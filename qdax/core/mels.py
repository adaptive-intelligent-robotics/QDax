"""Core components of the MAP-Elites Low-Spread algorithm."""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax

from qdax.core.containers.mels_repertoire import MELSRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.map_elites import MAPElites
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)
from qdax.utils.sampling import multi_sample_scoring_function


class MELS(MAPElites):
    """Core elements of the MAP-Elites Low-Spread algorithm.

    Most methods in this class are inherited from MAPElites.

    The same scoring function can be passed into both MAPElites and this class.
    We have overridden __init__ such that it takes in the scoring function and
    wraps it such that every solution is evaluated `num_samples` times.

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
        num_samples: int,
    ) -> None:
        self._scoring_function = partial(
            multi_sample_scoring_function,
            scoring_fn=scoring_function,
            num_samples=num_samples,
        )
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._num_samples = num_samples

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        genotypes: Genotype,
        centroids: Centroid,
        key: RNGKey,
    ) -> Tuple[MELSRepertoire, Optional[EmitterState], Metrics]:
        """Initialize a MAP-Elites Low-Spread repertoire with an initial
        population of genotypes. Requires the definition of centroids that can
        be computed with any method such as CVT or Euclidean mapping.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tessellation centroids of shape (batch_size, num_descriptors)
            key: a random key used for stochastic operations.

        Returns:
            A tuple of (initialized MAP-Elites Low-Spread repertoire, initial emitter
            state, JAX random key).
        """
        # score initial genotypes
        key, subkey = jax.random.split(key)
        fitnesses, descriptors, extra_scores = self._scoring_function(genotypes, subkey)

        # init the repertoire
        repertoire = MELSRepertoire.init(
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

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # calculate the initial metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics
