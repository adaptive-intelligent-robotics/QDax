"""Core components of the MAP-Elites Low-Spread algorithm."""
from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax

from qdax.core.containers.mels_repertoire import MELSRepertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.core.map_elites import MAPElites
from qdax.types import Centroid, Genotype, RNGKey


class MELS(MAPElites):
    """Core elements of the MAP-Elites Low-Spread algorithm.

    Note: most functions are inherited from MAPElites. The only function
    that had to be overwritten is the init function as it has to use the MELSRepertoire
    instead of MapElitesRepertoire.
    """

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
