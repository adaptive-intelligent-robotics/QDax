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
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class CMAMEState(EmitterState):
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
    emit_count: int


# TODO: add the pool of emitter - select the one with least emissions
# no pool in CMA-MEGA - did they realize it was not necessary?

# TODO: in paper pseudo-code, only indiv that have been added are used to update
# the distribution. Is it a mistake from the pseudo code or is it the desired
# behavior?

# TODO: is there a prioritizing of new cells before fitness improvement
# I think yes!!!
# CMA MEGA should be updated as well i think!

# TODO: refactoring

# TODO: think about the naming


class CMAMEImprovementEmitter(Emitter):
    def __init__(
        self,
        batch_size: int,
        genotype_dim: int,
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

        self._cma_initial_state = self._cmaes.init()

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[CMAMEState, RNGKey]:
        """
        Initializes the CMA-MEGA emitter


        Args:
            init_genotypes: initial genotypes to add to the grid.
            random_key: a random key to handle stochastic operations.

        Returns:
            The initial state of the emitter.
        """

        # return the initial state
        random_key, subkey = jax.random.split(random_key)
        return (
            CMAMEState(
                cmaes_state=self._cma_initial_state,
                random_key=subkey,
                emit_count=0,
            ),
            random_key,
        )

    @partial(jax.jit, static_argnames=("self", "batch_size"))
    def emit(
        self,
        repertoire: Optional[MapElitesRepertoire],
        emitter_state: CMAMEState,
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
        emitter_state: CMAMEState,
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
        improvements = fitnesses - repertoire.fitnesses[indices]

        sorting_criteria = self._sorting_criteria(
            fitnesses=fitnesses,
            descriptors=descriptors,
            repertoire=repertoire,
            improvements=improvements,
        )

        sorted_indices = jnp.argsort(sorting_criteria)[::-1]

        # Update CMA Parameters
        # TODO: only works for array atm
        sorted_candidates = genotypes[sorted_indices]
        cmaes_state = self._cmaes.update_state(cmaes_state, sorted_candidates)

        # If no improvement draw randomly and re-initialize parameters
        emit_count = emitter_state.emit_count + fitnesses.shape[0]
        reinitialize = jnp.all(improvements < 0) * (emit_count > self._min_count)

        # re-sample
        random_genotype, random_key = repertoire.sample(emitter_state.random_key, 1)

        # TODO: this should be hidden in the CMAES as a method
        # TODO: this is very ugly here

        # update - new state or init state if reinitialize is 1
        mean = random_genotype * reinitialize + jnp.nan_to_num(cmaes_state.mean) * (
            1 - reinitialize
        )
        cov = self._cma_initial_state.cov_matrix * reinitialize + jnp.nan_to_num(
            cmaes_state.cov_matrix
        ) * (1 - reinitialize)
        p_c = self._cma_initial_state.p_c * reinitialize + jnp.nan_to_num(
            cmaes_state.p_c
        ) * (1 - reinitialize)
        p_s = self._cma_initial_state.p_s * reinitialize + jnp.nan_to_num(
            cmaes_state.p_s
        ) * (1 - reinitialize)
        step_size = self._cma_initial_state.step_size * reinitialize + jnp.nan_to_num(
            cmaes_state.step_size
        ) * (1 - reinitialize)
        num_updates = 1 * reinitialize + cmaes_state.num_updates * (1 - reinitialize)

        # define new cmaes state
        cmaes_state = CMAESState(
            mean=mean,
            cov_matrix=cov,
            p_c=p_c,
            p_s=p_s,
            step_size=step_size,
            num_updates=num_updates,
        )

        # create new emitter state
        emitter_state = CMAMEState(
            cmaes_state=cmaes_state, random_key=random_key, emit_count=emit_count
        )

        return emitter_state

    def _sorting_criteria(self, fitnesses, descriptors, repertoire, improvements):
        """Defines how the genotypes should be sorted. Imapcts the update
        of the CMAES state. In the end, this defines the type of CMAES emitter
        used (optimizing, random direction or improvement)."""

        return improvements


class CMAMEOptimizingEmitter(CMAMEImprovementEmitter):
    def _sorting_criteria(self, fitnesses, descriptors, repertoire, improvements):
        """Defines how the genotypes should be sorted. Imapcts the update
        of the CMAES state. In the end, this defines the type of CMAES emitter
        used (optimizing, random direction or improvement)."""

        return fitnesses


class CMAMERandomDirectionEmitter(CMAMEImprovementEmitter):
    def _sorting_criteria(self, fitnesses, descriptors, repertoire, improvements):
        """Defines how the genotypes should be sorted. Imapcts the update
        of the CMAES state. In the end, this defines the type of CMAES emitter
        used (optimizing, random direction or improvement)."""

        return None
