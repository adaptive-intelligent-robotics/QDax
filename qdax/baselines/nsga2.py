"""Core components of the NSGA2 algorithm.

Link to paper: https://ieeexplore.ieee.org/document/996017
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax

from qdax.baselines.genetic_algorithm import GeneticAlgorithm
from qdax.core.containers.nsga2_repertoire import NSGA2Repertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.custom_types import Genotype, Metrics, RNGKey


class NSGA2(GeneticAlgorithm):
    """Implements main functions of the NSGA2 algorithm.

    This class inherits most functions from GeneticAlgorithm.
    The init function is overwritten in order to precise the type
    of repertoire used in NSGA2.

    Link to paper: https://ieeexplore.ieee.org/document/996017
    """

    @partial(jax.jit, static_argnames=("self", "population_size"))
    def init(
        self, genotypes: Genotype, population_size: int, key: RNGKey
    ) -> Tuple[NSGA2Repertoire, Optional[EmitterState], Metrics]:

        # score initial genotypes
        key, subkey = jax.random.split(key)
        fitnesses, extra_scores = self._scoring_function(genotypes, subkey)

        # init the repertoire
        repertoire = NSGA2Repertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            population_size=population_size,
        )

        # get initial state of the emitter
        key, subkey = jax.random.split(key)
        emitter_state = self._emitter.init(
            key=subkey,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=None,
            extra_scores=extra_scores,
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            extra_scores=extra_scores,
        )

        # calculate the initial metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics
