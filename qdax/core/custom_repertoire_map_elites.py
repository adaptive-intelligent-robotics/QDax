"""Core components of the MAP-Elites algorithm."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax

from qdax.core.map_elites import MAPElites

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire

from qdax.core.emitters.emitter import Emitter, EmitterState

from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)


class CustomRepertoireMAPElites(MAPElites):
    """Core elements of the MAP-Elites algorithm, with the ability to use custom repertoires.

    Args:
        scoring_function: a function that takes a batch of genotypes and compute
            their fitnesses and descriptors
        emitter: an emitter is used to suggest offsprings given a MAPELites
            repertoire. It has two compulsory functions. A function that takes
            emits a new population, and a function that update the internal state
            of the emitter.
        metrics_function: a function that takes a MAP-Elites repertoire and compute
            any useful metric to track its evolution
        repertoire_init: a function that initializes the MAP-Elites repertoire
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
        repertoire_init: Callable[
            [Genotype, Fitness, Descriptor, Centroid, ExtraScores], MapElitesRepertoire
        ] = MapElitesRepertoire.init,
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._repertoire_init = repertoire_init

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        genotypes: Genotype,
        centroids: Centroid,
        key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState]]:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tessellation centroids of shape (batch_size, num_descriptors)
            key: a random key used for stochastic operations.

        Returns:
            An initialized MAP-Elite repertoire with the initial state of the emitter,
            and a random key.
        """
        # score initial genotypes
        key, subkey = jax.random.split(key)
        fitnesses, descriptors, extra_scores = self._scoring_function(genotypes, subkey)

        # init the repertoire
        repertoire = self._repertoire_init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
        )

        # get initial state of the emitter
        key, subkey = jax.random.split(key)
        emitter_state = self._emitter.init(
            key=subkey,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return repertoire, emitter_state
