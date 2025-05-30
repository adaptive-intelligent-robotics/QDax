from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from qdax.baselines.cmaes import CMAESState
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.cma_emitter import CMAEmitter, CMAEmitterState
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class CMARndEmitterState(CMAEmitterState):
    """
    Emitter state for the CMA-ME random direction emitter.


    Args:
        key: a random key to handle stochastic operations. Used for
            state update only, another key is used to emit. This might be
            subject to refactoring discussions in the future.
        cmaes_state: state of the underlying CMA-ES algorithm
        previous_fitnesses: store last fitnesses of the repertoire. Used to
            compute the improvement.
        emit_count: count the number of emission events.
        random_direction: direction of the descriptor space we are trying to
            explore.
    """

    random_direction: Descriptor


class CMARndEmitter(CMAEmitter):
    def init(
        self,
        key: RNGKey,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> CMARndEmitterState:
        """
        Initializes the CMA-MEGA emitter


        Args:
            genotypes: initial genotypes to add to the grid.
            key: a random key to handle stochastic operations.

        Returns:
            The initial state of the emitter.
        """

        # Initialize repertoire with default values
        num_centroids = self._centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=(num_centroids, 1))

        # take a random direction
        key, subkey = jax.random.split(key)
        random_direction = jax.random.uniform(
            subkey,
            shape=(self._centroids.shape[-1],),
        )

        # return the initial state
        key, subkey = jax.random.split(key)

        emitter_state = CMARndEmitterState(
            key=subkey,
            cmaes_state=self._cma_initial_state,
            previous_fitnesses=default_fitnesses,
            emit_count=0,
            random_direction=random_direction,
        )

        return emitter_state

    def _update_and_init_emitter_state(
        self,
        cmaes_state: CMAESState,
        emitter_state: CMAEmitterState,
        repertoire: MapElitesRepertoire,
        emit_count: int,
        key: RNGKey,
    ) -> CMAEmitterState:
        """Update the emitter state in the case of a reinit event.
        Reinit the cmaes state and use an individual from the repertoire
        as the starting mean.

        Args:
            cmaes_state: current cmaes state
            emitter_state: current cmame state
            repertoire: most recent repertoire
            emit_count: counter of the emitter
            key: key to handle stochastic events

        Returns:
            The updated emitter state.
        """

        # re-sample
        key, subkey = jax.random.split(key)
        random_genotype = repertoire.select(
            subkey, num_samples=1, selector=self._selector
        ).genotypes

        # get new mean - remove the batch dim
        new_mean = jax.tree.map(lambda x: x.squeeze(0), random_genotype)

        # define the corresponding cmaes init state
        cmaes_init_state = self._cma_initial_state.replace(mean=new_mean, num_updates=0)

        # take a new random direction
        random_direction = jax.random.uniform(
            key,
            shape=(self._centroids.shape[-1],),
        )

        emitter_state = emitter_state.replace(
            cmaes_state=cmaes_init_state,
            emit_count=0,
            random_direction=random_direction,
        )

        return emitter_state  # type: ignore

    def _ranking_criteria(
        self,
        emitter_state: CMARndEmitterState,
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
            improvements: improvements of the emitted genotypes. This corresponds
                to the difference between their fitness and the fitness of the
                individual occupying the cell of corresponding fitness.

        Returns:
            The values to take into account in order to rank the emitted genotypes.
            Here, it is the dot product of the descriptor with the current random
            direction.
        """

        # criteria: projection of the descriptors along the random direction
        ranking_criteria = jnp.dot(descriptors, emitter_state.random_direction)

        # make sure to have all the new cells first
        new_cell_offset = jnp.max(ranking_criteria) - jnp.min(ranking_criteria)

        # condition for being a new cell
        condition = improvements == jnp.inf

        ranking_criteria = jnp.where(
            condition, ranking_criteria + new_cell_offset, ranking_criteria
        )

        return ranking_criteria  # type: ignore
