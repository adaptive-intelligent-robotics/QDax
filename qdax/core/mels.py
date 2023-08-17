"""Core components of the MAP-Elites Low-Spread algorithm."""
from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mels_repertoire import MELSRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.map_elites import MAPElites
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)


class MELS(MAPElites):
    """Core elements of the MAP-Elites Low-Spread algorithm.

    Most methods in this class are inherited from MAPElites.

    The same scoring function can be passed into both MAPElites and this class.
    We have overridden __init__ such that it takes in the scoring function and
    wraps it such that every solution is evaluated `num_evals` times.

    We also overrode the init method to use the MELSRepertoire instead of
    MapElitesRepertoire.
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MELSRepertoire], Metrics],
        num_evals: int,
    ) -> None:
        def multi_eval_scoring_function(
            genotypes: Genotype, random_key: RNGKey
        ) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
            """Evaluates the given genotypes `num_evals` times."""
            subkeys = jax.random.split(random_key, num_evals + 1)
            random_key = subkeys[0]
            fitnesses, descriptors, extra_scores, _ = jax.vmap(
                scoring_function, in_axes=(None, 0)
            )(genotypes, subkeys[1:])

            # The vmap will output arrays with shape (num_evals, batch_size, ...)
            # but we need to have (batch_size, num_evals, ...) for MELSRepertoire.
            fitnesses = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), fitnesses)
            descriptors = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), descriptors)
            extra_scores = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), extra_scores)

            return fitnesses, descriptors, extra_scores, random_key

        self._scoring_function = multi_eval_scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._num_evals = num_evals

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        init_genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[MELSRepertoire, Optional[EmitterState], RNGKey]:
        """
        Initialize a MAP-Elites Low-Spread repertoire with an initial population of
        genotypes. Requires the definition of centroids that can be computed with any
        method such as CVT or Euclidean mapping.

        Args:
            init_genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tessellation centroids of shape (batch_size, num_descriptors)
            random_key: a random key used for stochastic operations.

        Returns:
            A tuple of (initialized MAP-Elites Low-Spread repertoire, initial emitter
            state, JAX random key).
        """
        # score initial genotypes
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            init_genotypes, random_key
        )

        # init the repertoire
        repertoire = MELSRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return repertoire, emitter_state, random_key
