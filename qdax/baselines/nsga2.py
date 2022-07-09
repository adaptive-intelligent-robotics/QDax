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
from qdax.types import Genotype, RNGKey


class NSGA2(GeneticAlgorithm):
    """Implements main functions of the NSGA2 algorithm.

    This class inherits most functions from GeneticAlgorithm.
    The init function is overwritten in order to precise the type
    of repertoire used in NSGA2.

    Link to paper: https://ieeexplore.ieee.org/document/996017
    """

    @partial(jax.jit, static_argnames=("self", "population_size"))
    def init(
        self, init_genotypes: Genotype, population_size: int, random_key: RNGKey
    ) -> Tuple[NSGA2Repertoire, Optional[EmitterState], RNGKey]:

        # score initial genotypes
        fitnesses, extra_scores, random_key = self._scoring_function(
            init_genotypes, random_key
        )

        # init the repertoire
        repertoire = NSGA2Repertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            population_size=population_size,
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
            extra_scores=extra_scores,
        )

        return repertoire, emitter_state, random_key
