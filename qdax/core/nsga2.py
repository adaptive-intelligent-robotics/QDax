from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax

from qdax.core.containers.nsga2_repertoire import NSGA2Repertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.core.genetic_algorithm import GeneticAlgorithm
from qdax.types import Genotype, RNGKey


class NSGA2(GeneticAlgorithm):
    """Implements main functions of the NSGA2 algorithm.


    TODO: add link to paper.
    """

    @partial(jax.jit, static_argnames=("self", "population_size"))
    def init(
        self, init_genotypes: Genotype, population_size: int, random_key: RNGKey
    ) -> Tuple[NSGA2Repertoire, Optional[EmitterState], RNGKey]:

        # score initial population
        fitnesses = self._scoring_function(init_genotypes)

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

        return repertoire, emitter_state, random_key
