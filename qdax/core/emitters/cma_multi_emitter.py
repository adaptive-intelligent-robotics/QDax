from __future__ import annotations

from functools import partial
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.cma_emitter import CMAEmitter, CMAEmitterState
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

# TODO: change name for CMAPoolEmitter


class CMAMultiEmitterState(EmitterState):
    """
    Emitter state for the CMA-ME emitter.

    Args:
        random_key: a random key to handle stochastic operations. Used for
            state update only, another key is used to emit. This might be
            subject to refactoring discussions in the future.
        cmaes_state: state of the underlying CMA-ES algorithm
    """

    current_index: int
    emitter_states: CMAEmitterState


class CMAMultiEmitter(Emitter):
    def __init__(self, num_states: int, emitter: CMAEmitter):
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
        self._num_states = num_states
        self._emitter = emitter

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

        def scan_emitter_init(
            carry: RNGKey, unused: Any
        ) -> Tuple[RNGKey, CMAEmitterState]:
            random_key = carry
            emitter_state, random_key = self._emitter.init(init_genotypes, random_key)
            return random_key, emitter_state

        random_key, emitter_states = jax.lax.scan(
            scan_emitter_init, random_key, (), length=self._num_states
        )

        emitter_state = CMAMultiEmitterState(
            current_index=0, emitter_states=emitter_states
        )

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
        current_index = emitter_state.current_index
        used_emitter_state = jax.tree_util.tree_map(
            lambda x: x[current_index], emitter_state.emitter_states
        )

        offsprings, random_key = self._emitter.emit(
            repertoire, used_emitter_state, random_key
        )

        return offsprings, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: CMAMultiEmitterState,
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

        # retrieve the emitter that has been used and it's emitter state
        current_index = emitter_state.current_index
        emitter_states = emitter_state.emitter_states

        used_emitter_state = jax.tree_util.tree_map(
            lambda x: x[current_index], emitter_states
        )

        # update the used emitter state
        used_emitter_state = self._emitter.state_update(
            emitter_state=used_emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update the emitter state
        emitter_states = jax.tree_util.tree_map(
            lambda x, y: x.at[current_index].set(y), emitter_states, used_emitter_state
        )

        # determine the next emitter to be used
        emit_counts = emitter_states.emit_count

        new_index = jnp.argmin(emit_counts)

        emitter_state = emitter_state.replace(
            current_index=new_index, emitter_states=emitter_states
        )

        return emitter_state
