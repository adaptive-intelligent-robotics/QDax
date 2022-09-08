from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from qdax.core.cmaes import CMAES, CMAESState
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class CMAEmitterState(EmitterState):
    """
    Emitter state for the CMA-ME emitter.

    Args:
        random_key: a random key to handle stochastic operations. Used for
            state update only, another key is used to emit. This might be
            subject to refactoring discussions in the future.
        cmaes_state: state of the underlying CMA-ES algorithm
    """

    random_key: RNGKey
    cmaes_state: CMAESState
    previous_repertoire: MapElitesRepertoire
    emit_count: int


# TODO: wait for confirmation before doing so.
# TODO: current implem not adapted to pool of emitter
# TODO: we should need emitters and schedulers for clean implem
# TODO: add the pool of emitter - select the one with least emissions
# no pool in CMA-MEGA - did they realize it was not necessary?

# TODO: in paper pseudo-code, only indiv that have been added are used to update
# the distribution. Is it a mistake from the pseudo code or is it the desired
# behavior? - for some emitters

# TODO: should we have an option in cmaes to update the weights?

# TODO: is there a prioritizing of new cells before fitness improvement
# I think yes!!!
# CMA MEGA should be updated as well i think!

# among the new cells, are they prioritized based on fitness or not?

# TODO: I want to introduce a init_void in MAPElitesRepertoire
# it could be used at least three time in the package

# TODO: make sure my decision to have the improvement emitter the default one
# and not precised in its name - is ok for everyone

# TODO: shouldn't we start with num_updates=0 ???


class CMAEmitter(Emitter):
    def __init__(
        self,
        batch_size: int,
        genotype_dim: int,
        centroids: Centroid,
        sigma_g: float,
        step_size: Optional[float] = None,
        min_count: Optional[int] = None,
    ):
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
        self._weights = jnp.expand_dims(
            jnp.log(batch_size + 0.5) - jnp.log(jnp.arange(1, batch_size + 1)), axis=-1
        )
        self._weights = self._weights / (self._weights.sum())

        if step_size is None:
            step_size = 1.0

        # define a CMAES instance
        self._cmaes = CMAES(
            population_size=batch_size,
            search_dim=genotype_dim,
            # no need for fitness function in that specific case
            fitness_function=None,  # type: ignore
            num_best=batch_size,
            init_sigma=sigma_g,
            init_step_size=step_size,
            bias_weights=True,
        )

        # minimum number of emitted solution before an emitter can be re-initialized
        if min_count is None:
            min_count = 0

        self._min_count = min_count

        self._centroids = centroids

        self._cma_initial_state = self._cmaes.init()

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[CMAEmitterState, RNGKey]:
        """
        Initializes the CMA-MEGA emitter


        Args:
            init_genotypes: initial genotypes to add to the grid.
            random_key: a random key to handle stochastic operations.

        Returns:
            The initial state of the emitter.
        """

        # Initialize repertoire with default values
        num_centroids = self._centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)
        default_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape[1:]),
            init_genotypes,
        )
        default_descriptors = jnp.zeros(
            shape=(num_centroids, self._centroids.shape[-1])
        )

        repertoire = MapElitesRepertoire(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=self._centroids,
        )

        # return the initial state
        random_key, subkey = jax.random.split(random_key)
        return (
            CMAEmitterState(
                random_key=subkey,
                cmaes_state=self._cma_initial_state,
                previous_repertoire=repertoire,
                emit_count=0,
            ),
            random_key,
        )

    @partial(jax.jit, static_argnames=("self", "batch_size"))
    def emit(
        self,
        repertoire: Optional[MapElitesRepertoire],
        emitter_state: CMAEmitterState,
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
        # emit from CMA-ES
        offsprings, _ = self._cmaes.sample(
            cmaes_state=emitter_state.cmaes_state, random_key=emitter_state.random_key
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

        # Update CMA Parameters
        cmaes_state = self._cmaes.update_state(cmaes_state, sorted_candidates)

        # If no improvement draw randomly and re-initialize parameters
        emit_count = emitter_state.emit_count + fitnesses.shape[0]
        reinitialize = jnp.all(improvements < 0) * (emit_count > self._min_count)

        def update_and_reinit(
            operand: Tuple[
                CMAESState, CMAEmitterState, MapElitesRepertoire, int, RNGKey
            ],
        ) -> Tuple[CMAEmitterState, RNGKey]:
            return self._update_and_init_emitter_state(*operand)

        def update_wo_reinit(
            operand: Tuple[
                CMAESState, CMAEmitterState, MapElitesRepertoire, int, RNGKey
            ],
        ) -> Tuple[CMAEmitterState, RNGKey]:

            (cmaes_state, emitter_state, repertoire, emit_count, random_key) = operand

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
