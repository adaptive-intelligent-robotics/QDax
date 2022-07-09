"""Core components of the SPEA2 algorithm.

Link to paper: "https://www.semanticscholar.org/paper/SPEA2%3A
-Improving-the-strength-pareto-evolutionary-Zitzler-Laumanns/
b13724cb54ae4171916f3f969d304b9e9752a57f"
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax

from qdax.baselines.genetic_algorithm import GeneticAlgorithm
from qdax.core.containers.spea2_repertoire import SPEA2Repertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.types import Genotype, RNGKey


class SPEA2(GeneticAlgorithm):
    """Implements main functions of the SPEA2 algorithm.

    This class inherits most functions from GeneticAlgorithm.
    The init function is overwritten in order to precise the type
    of repertoire used in SPEA2.

    Link to paper: "https://www.semanticscholar.org/paper/SPEA2%3A-
    Improving-the-strength-pareto-evolutionary-Zitzler-Laumanns/
    b13724cb54ae4171916f3f969d304b9e9752a57f"
    """

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "population_size",
            "num_neighbours",
        ),
    )
    def init(
        self,
        init_genotypes: Genotype,
        population_size: int,
        num_neighbours: int,
        random_key: RNGKey,
    ) -> Tuple[SPEA2Repertoire, Optional[EmitterState], RNGKey]:

        # score initial genotypes
        fitnesses, extra_scores, random_key = self._scoring_function(
            init_genotypes, random_key
        )

        # init the repertoire
        repertoire = SPEA2Repertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            population_size=population_size,
            num_neighbours=num_neighbours,
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
