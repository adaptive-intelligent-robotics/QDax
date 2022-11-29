from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.core.map_elites import MAPElites
from qdax.types import Centroid, RNGKey


class MOME(MAPElites):
    """Implements Multi-Objectives MAP Elites.

    Note: most functions are inherited from MAPElites. The only function
    that had to be overwritten is the init function as it has to take
    into account the specificities of the the Multi Objective repertoire.
    """

    @partial(jax.jit, static_argnames=("self", "pareto_front_max_length"))
    def init(
        self,
        init_genotypes: jnp.ndarray,
        centroids: Centroid,
        pareto_front_max_length: int,
        random_key: RNGKey,
    ) -> Tuple[MOMERepertoire, Optional[EmitterState], RNGKey]:
        """Initialize a MOME grid with an initial population of genotypes. Requires
        the definition of centroids that can be computed with any method such as
        CVT or Euclidean mapping.

        Args:
            init_genotypes: genotypes of the initial population.
            centroids: centroids of the repertoire.
            pareto_front_max_length: maximum size of the pareto front. This is
                necessary to respect jax.jit fixed shape size constraint.
            random_key: a random key to handle stochasticity.

        Returns:
            The initial repertoire and emitter state, and a new random key.
        """

        # first score
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            init_genotypes, random_key
        )

        # init the repertoire
        repertoire = MOMERepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            pareto_front_max_length=pareto_front_max_length,
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
