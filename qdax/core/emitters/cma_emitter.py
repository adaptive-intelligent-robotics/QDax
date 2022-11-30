from __future__ import annotations

from abc import ABC, abstractmethod
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
        previous_fitnesses: store last fitnesses of the repertoire. Used to
            compute the improvment.
        emit_count: count the number of emission events.
    """

    random_key: RNGKey
    cmaes_state: CMAESState
    previous_fitnesses: Fitness
    emit_count: int


class CMAEmitter(Emitter, ABC):
    def __init__(
        self,
        batch_size: int,
        genotype_dim: int,
        centroids: Centroid,
        sigma_g: float,
        min_count: Optional[int] = None,
        max_count: Optional[float] = None,
    ):
        """
        Class for the emitter of CMA ME from "Covariance Matrix Adaptation for the
        Rapid Illumination of Behavior Space" by Fontaine et al.

        Args:
            batch_size: number of solutions sampled at each iteration
            genotype_dim: dimension of the genotype space.
            centroids: centroids used for the repertoire.
            sigma_g: standard deviation for the coefficients - called step size.
            min_count: minimum number of CMAES opt step before being considered for
                reinitialisation.
            max_count: maximum number of CMAES opt step authorized.
        """
        self._batch_size = batch_size

        # define a CMAES instance
        self._cmaes = CMAES(
            population_size=batch_size,
            search_dim=genotype_dim,
            # no need for fitness function in that specific case
            fitness_function=None,  # type: ignore
            num_best=batch_size,
            init_sigma=sigma_g,
            mean_init=None,  # will be init at zeros in cmaes
            bias_weights=True,
            delay_eigen_decomposition=True,
        )

        # minimum number of emitted solution before an emitter can be re-initialized
        if min_count is None:
            min_count = 0

        self._min_count = min_count

        if max_count is None:
            max_count = jnp.inf

        self._max_count = max_count

        self._centroids = centroids

        self._cma_initial_state = self._cmaes.init()

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._batch_size

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

        # return the initial state
        random_key, subkey = jax.random.split(random_key)
        return (
            CMAEmitterState(
                random_key=subkey,
                cmaes_state=self._cma_initial_state,
                previous_fitnesses=default_fitnesses,
                emit_count=0,
            ),
            random_key,
        )

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Optional[MapElitesRepertoire],
        emitter_state: CMAEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        Emits new individuals. Interestingly, this method does not directly modifies
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
        offsprings, random_key = self._cmaes.sample(
            cmaes_state=emitter_state.cmaes_state, random_key=random_key
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
        Updates the CMA-ME emitter state.

        Note: we use the update_state function from CMAES, a function that assumes
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
        improvements = fitnesses - emitter_state.previous_fitnesses[indices]

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

        # compute reinitialize condition
        emit_count = emitter_state.emit_count + 1

        # check if the criteria are too similar
        sorted_criteria = ranking_criteria[sorted_indices]
        flat_criteria_condition = (
            jnp.linalg.norm(sorted_criteria[0] - sorted_criteria[-1]) < 1e-12
        )

        # check all conditions
        reinitialize = (
            jnp.all(improvements < 0) * (emit_count > self._min_count)
            + (emit_count > self._max_count)
            + self._cmaes.stop_condition(cmaes_state)
            + flat_criteria_condition
        )

        # If true, draw randomly and re-initialize parameters
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
            """Update the emitter when no reinit event happened.

            Here lies a divergence compared to the original implementation. We
            are getting better results when using no mask and doing the update
            with the whole batch of individuals rather than keeping only the one
            than were added to the archive.

            Interestingly, keeping the best half was not doing better. We think that
            this might be due to the small batch size used.

            This applies for the setting from the paper CMA-ME. Those facts might
            not be true with other problems and hyperparameters.

            To replicate the code described in the paper, replace:
            `mask = jnp.ones_like(sorted_improvements)`

            by:
            ```
            mask = sorted_improvements >= 0
            mask = mask + 1e-6
            ```

            RMQ: the addition of 1e-6 is here to fix a numerical
            instability.
            """

            (cmaes_state, emitter_state, repertoire, emit_count, random_key) = operand

            # Update CMA Parameters
            mask = jnp.ones_like(sorted_improvements)

            cmaes_state = self._cmaes.update_state_with_mask(
                cmaes_state, sorted_candidates, mask=mask
            )

            emitter_state = emitter_state.replace(
                cmaes_state=cmaes_state,
                emit_count=emit_count,
            )

            return emitter_state, random_key

        # Update CMA Parameters
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

        # update the emitter state
        emitter_state = emitter_state.replace(
            random_key=random_key, previous_fitnesses=repertoire.fitnesses
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
        """Update the emitter state in the case of a reinit event.
        Reinit the cmaes state and use an individual from the repertoire
        as the starting mean.

        Args:
            cmaes_state: current cmaes state
            emitter_state: current cmame state
            repertoire: most recent repertoire
            emit_count: counter of the emitter
            random_key: key to handle stochastic events

        Returns:
            The updated emitter state.
        """

        # re-sample
        random_genotype, random_key = repertoire.sample(random_key, 1)

        # remove the batch dim
        new_mean = jax.tree_util.tree_map(lambda x: x.squeeze(0), random_genotype)

        cmaes_init_state = self._cma_initial_state.replace(mean=new_mean, num_updates=0)

        emitter_state = emitter_state.replace(
            cmaes_state=cmaes_init_state, emit_count=0
        )

        return emitter_state, random_key

    @abstractmethod
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
        """Defines how the genotypes should be sorted. Impacts the update
        of the CMAES state. In the end, this defines the type of CMAES emitter
        used (optimizing, random direction or improvement).

        Args:
            emitter_state: current state of the emitter.
            repertoire: latest repertoire of genotypes.
            genotypes: emitted genotypes.
            fitnesses: corresponding fitnesses.
            descriptors: corresponding fitnesses.
            extra_scores: corresponding extra scores.
            improvements: improvments of the emitted genotypes. This corresponds
                to the difference between their fitness and the fitness of the
                individual occupying the cell of corresponding fitness.

        Returns:
            The values to take into account in order to rank the emitted genotypes.
            Here, it's the improvement, or the fitness when the cell was previously
            unoccupied. Additionally, genotypes that discovered a new cell are
            given on offset to be ranked in front of other genotypes.
        """

        pass
