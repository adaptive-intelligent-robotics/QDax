from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mome_repertoire import MOMERepertoire
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

class MOME(MAPElites):
    """Implements Multi-Objectives MAP Elites.

    Note: most functions are inherited from MAPElites. The only function
    that had to be overwritten is the init function as it has to take
    into account the specificities of the the Multi Objective repertoire.
    """
    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MOMERepertoire], Metrics],
        repertoire_init: Callable[
            [Genotype, Fitness, Descriptor, Centroid, ExtraScores], MOMERepertoire
        ] = MOMERepertoire.init,
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._repertoire_init = repertoire_init


    @partial(jax.jit, static_argnames=("self", "pareto_front_max_length"))
    def init(
        self,
        genotypes: jnp.ndarray,
        centroids: Centroid,
        pareto_front_max_length: int,
        key: RNGKey,
    ) -> Tuple[MOMERepertoire, Optional[EmitterState]]:
        """Initialize a MOME grid with an initial population of genotypes. Requires
        the definition of centroids that can be computed with any method such as
        CVT or Euclidean mapping.

        Args:
            genotypes: genotypes of the initial population.
            centroids: centroids of the repertoire.
            pareto_front_max_length: maximum size of the pareto front. This is
                necessary to respect jax.jit fixed shape size constraint.
            key: a random key to handle stochasticity.

        Returns:
            The initial repertoire and emitter state, and a new random key.
        """
        # score initial genotypes
        key, subkey = jax.random.split(key)
        fitnesses, descriptors, extra_scores = self._scoring_function(genotypes, subkey)

        return self.init_ask_tell(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            pareto_front_max_length=pareto_front_max_length,
            key=key,
            extra_scores=extra_scores,
        )


    @partial(jax.jit, static_argnames=("self", "pareto_front_max_length"))
    def init_ask_tell(
        self,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        pareto_front_max_length: int,
        key: RNGKey,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Tuple[MOMERepertoire, Optional[EmitterState]]:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes and their evaluations. 
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            fitnesses: initial fitnesses of the genotypes
            descriptors: initial descriptors of the genotypes
            centroids: tessellation centroids of shape (batch_size, num_descriptors)
            pareto_front_max_length: maximum size of the pareto front. This is
                necessary to respect jax.jit fixed shape size constraint.
            key: a random key used for stochastic operations.
            extra_scores: extra scores of the initial genotypes (optional)

        Returns:
            An initialized MAP-Elite repertoire with the initial state of the emitter.
        """
        if extra_scores is None:
            extra_scores = {}
        # init the repertoire
        repertoire = self._repertoire_init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
            pareto_front_max_length=pareto_front_max_length
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