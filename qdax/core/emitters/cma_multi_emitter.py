from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp

from qdax.core.cmaes import CMAES, CMAESState
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax.core.emitters.cma_emitter import CMAEmitter, CMAEmitterState
from qdax.core.emitters.cma_opt_emitter import CMAOptimizingEmitter
from qdax.core.emitters.cma_rnd_emitter import CMARndEmitter, CMARndEmitterState
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class CMAMultiEmitterState(EmitterState):
    """
    Emitter state for the CMA-ME emitter.

    Args:
        random_key: a random key to handle stochastic operations. Used for
            state update only, another key is used to emit. This might be
            subject to refactoring discussions in the future.
        cmaes_state: state of the underlying CMA-ES algorithm
    """

    emitter_states: Tuple[Union[CMAEmitterState, CMARndEmitterState], ...]


class CMAMultiEmitter(Emitter):
    def __init__(self, emitters: Tuple[Union[CMAEmitter, CMARndEmitter], ...]):
        """
        Class for the emitter of CMA ME from "Covariance Matrix Adaptation for the
        Rapid Illumination of Behavior Space" by Fontaine et al.

        Args:
            batch_size: number of solutions sampled at each iteration
            learning_rate: rate at which the mean of the distribution is updated.
            genotype_dim: dimension of the genotype space.
            sigma_g: standard deviation for the coefficients
            step_size: size of the steps used in CMAES updates
        """
        self._emitters = emitters

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[CMAMultiEmitterState, RNGKey]:
        """
        Initializes the CMA-MEGA emitter


        Args:
            init_genotypes: initial genotypes to add to the grid.
            random_key: a random key to handle stochastic operations.

        Returns:
            The initial state of the emitter.
        """
        emitter_states = []
        for emitter in self._emitters:
            emitter_state, random_key = emitter.init(init_genotypes, random_key)
            emitter_states.append(emitter_state)

        emitter_state = CMAMultiEmitterState(emitter_states=tuple(emitter_states))

        return (
            emitter_state,
            random_key,
        )

    @partial(jax.jit, static_argnames=("self", "batch_size"))
    def emit(
        self,
        repertoire: Optional[MapElitesRepertoire],
        emitter_state: CMAMultiEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        Emits new individuals. Interestingly, this method does not directly modify
        individuals from the repertoire but sample from a distribution. Hence the
        repertoire is not used in the emit function.

        Args:
            repertoire: a repertoire of genotypes (unused).
            emitter_state: the state of the CMA-MEGA emitter.
            random_key: a random key to handle random operations.

        Returns:
            New genotypes and a new random key.
        """
        emit_counts = [
            emitter_s.emit_count for emitter_s in emitter_state.emitter_states
        ]

        index = jnp.argmin(emit_counts)

        used_emitter = self._emitters[index]
        used_emitter_state = emitter_state.emitter_states[index]

        offsprings, random_key = used_emitter.emit(
            repertoire, used_emitter_state, random_key
        )

        return offsprings, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: CMAEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Optional[EmitterState]:
        """
        Updates the CMA-MEGA emitter state.

        Note: in order to recover the coeffs that where used to sample the genotypes,
        we reuse the emitter state's random key in this function.

        Note: we use the update_state function from CMAES, a function that suppose
        that the candidates are already sorted. We do this because we have to sort
        them in this function anyway, in order to apply the right weights to the
        terms when update theta.

        Args:
            emitter_state: current emitter state
            repertoire: the current genotypes repertoire
            genotypes: the genotypes of the batch of emitted offspring (unused).
            fitnesses: the fitnesses of the batch of emitted offspring.
            descriptors: the descriptors of the emitted offspring.
            extra_scores: unused

        Returns:
            The updated emitter state.
        """

        # retrieve elements from the emitter state
        cmaes_state = emitter_state.cmaes_state

        # Compute the improvements - needed for re-init condition
        indices = get_cells_indices(descriptors, repertoire.centroids)
        improvements = fitnesses - emitter_state.previous_repertoire.fitnesses[indices]

        ranking_criteria = self._ranking_criteria(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            improvements=improvements,
        )

        # get the indices
        sorted_indices = jnp.flip(jnp.argsort(ranking_criteria))

        # sort the candidates
        sorted_candidates = jax.tree_util.tree_map(
            lambda x: x[sorted_indices], genotypes
        )
        sorted_improvements = improvements[sorted_indices]

        # # Update CMA Parameters
        # cmaes_state = self._cmaes.update_state(
        #     cmaes_state, sorted_candidates, mask=(improvements >= 0)
        # )

        # If no improvement draw randomly and re-initialize parameters
        emit_count = emitter_state.emit_count + fitnesses.shape[0]
        reinitialize = jnp.all(improvements < 0) * (emit_count > self._min_count)

        def update_and_reinit(
            operand: Tuple[
                CMAESState, CMAEmitterState, MapElitesRepertoire, int, RNGKey
            ],
        ) -> Tuple[CMAEmitterState, RNGKey]:
            print("Reinit event happened.")
            return self._update_and_init_emitter_state(*operand)

        def update_wo_reinit(
            operand: Tuple[
                CMAESState, CMAEmitterState, MapElitesRepertoire, int, RNGKey
            ],
        ) -> Tuple[CMAEmitterState, RNGKey]:

            (cmaes_state, emitter_state, repertoire, emit_count, random_key) = operand

            # Update CMA Parameters
            mask = sorted_improvements >= 0
            # mask = jnp.ones_like(sorted_improvements) * (sorted_improvements >= 0)
            mask = mask + 0.000001

            # weights = jnp.log(
            #     (self._batch_size + 1)
            #     / jnp.arange(start=1, stop=(self._batch_size + 1))
            # )

            # print("Weights : ", weights)

            # weights = jnp.multiply(weights, mask)
            # weights = weights / (weights.sum())

            # print("Weights : ", weights)

            cmaes_state = self._cmaes.update_state_with_mask(
                cmaes_state, sorted_candidates, mask=mask
            )

            emitter_state = emitter_state.replace(
                cmaes_state=cmaes_state,
                emit_count=emit_count,
            )

            return emitter_state, random_key

        emitter_state, random_key = jax.lax.cond(
            reinitialize,
            update_and_reinit,
            update_wo_reinit,
            operand=(
                cmaes_state,
                emitter_state,
                repertoire,
                emit_count,
                emitter_state.random_key,
            ),
        )

        emitter_state = emitter_state.replace(
            random_key=random_key, previous_repertoire=repertoire
        )

        return emitter_state

    def _update_and_init_emitter_state(
        self,
        cmaes_state: CMAESState,
        emitter_state: CMAEmitterState,
        repertoire: MapElitesRepertoire,
        emit_count: int,
        random_key: RNGKey,
    ) -> Tuple[CMAEmitterState, RNGKey]:

        # re-sample
        random_genotype, random_key = repertoire.sample(random_key, 1)

        # remove the batch dim
        new_mean = jax.tree_util.tree_map(lambda x: x.squeeze(0), random_genotype)

        cmaes_init_state = self._cma_initial_state.replace(mean=new_mean, num_updates=1)

        emitter_state = emitter_state.replace(
            cmaes_state=cmaes_init_state, emit_count=0
        )

        return emitter_state, random_key

    def _ranking_criteria(
        self,
        emitter_state: CMAEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
        improvements: jnp.ndarray,
    ) -> jnp.ndarray:
        """Defines how the genotypes should be sorted. Imapcts the update
        of the CMAES state. In the end, this defines the type of CMAES emitter
        used (optimizing, random direction or improvement)."""

        # condition for being a new cell
        condition = improvements == jnp.inf

        # criteria: fitness if new cell, improvement else
        ranking_criteria = jnp.where(condition, x=fitnesses, y=improvements)

        # make sure to have all the new cells first
        new_cell_offset = jnp.max(ranking_criteria) - jnp.min(ranking_criteria)

        ranking_criteria = jnp.where(
            condition, x=ranking_criteria + new_cell_offset, y=ranking_criteria
        )

        return ranking_criteria  # type: ignore
