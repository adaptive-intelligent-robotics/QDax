from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.cmaes import CMAESState
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.cma_emitter import CMAEmitter, CMAEmitterState
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class CMARndEmitterState(CMAEmitterState):
    """
    Emitter state for the CMA-ME emitter.

    # TODO: update

    Args:
        random_key: a random key to handle stochastic operations. Used for
            state update only, another key is used to emit. This might be
            subject to refactoring discussions in the future.
        cmaes_state: state of the underlying CMA-ES algorithm
    """

    random_direction: Descriptor


class CMARndEmitter(CMAEmitter):
    @partial(jax.jit, static_argnames=("self",))
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[CMARndEmitterState, RNGKey]:
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

        # take a random direction
        random_key, subkey = jax.random.split(random_key)
        random_direction = jax.random.uniform(
            subkey,
            shape=(self._centroids.shape[-1],),
        )

        # return the initial state
        random_key, subkey = jax.random.split(random_key)

        return (
            CMARndEmitterState(
                random_key=subkey,
                cmaes_state=self._cma_initial_state,
                previous_repertoire=repertoire,
                emit_count=0,
                random_direction=random_direction,
            ),
            random_key,
        )

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

        # take a new random direction
        random_key, subkey = jax.random.split(random_key)
        random_direction = jax.random.uniform(
            subkey,
            shape=(self._centroids.shape[-1],),
        )

        emitter_state = emitter_state.replace(
            cmaes_state=cmaes_init_state,
            emit_count=0,
            random_direction=random_direction,
        )

        return emitter_state, random_key

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
        """Defines how the genotypes should be sorted. Imapcts the update
        of the CMAES state. In the end, this defines the type of CMAES emitter
        used (optimizing, random direction or improvement)."""

        # projection of the descriptors along the random direction
        return jnp.dot(descriptors, emitter_state.random_direction)
