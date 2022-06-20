from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax

from qdax.core.containers.population_repertoire import PopulationRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Fitness, Genotype, RNGKey


class GeneticAlgorithm:
    """"""

    def __init__(
        self,
        scoring_function: Callable[[Genotype], Fitness],
        emitter: Emitter,
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter

    @partial(jax.jit, static_argnames=("self", "population_size"))
    def init(
        self, init_genotypes: Genotype, population_size: int, random_key: RNGKey
    ) -> Tuple[PopulationRepertoire, Optional[EmitterState], RNGKey]:

        raise NotImplementedError

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: PopulationRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[PopulationRepertoire, Optional[EmitterState], RNGKey]:

        # generate offsprings
        genotypes, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        # score the offsprings
        fitnesses = self._scoring_function(genotypes)

        # update the repertoire
        repertoire = repertoire.add(genotypes, fitnesses)

        return repertoire, emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def scan_update(
        self,
        carry: Tuple[PopulationRepertoire, Optional[EmitterState], RNGKey],
        unused: Any,
    ) -> Tuple[Tuple[PopulationRepertoire, Optional[EmitterState], RNGKey], Any]:
        # iterate over grid
        repertoire, emitter_state, random_key = carry
        repertoire, emitter_state, random_key = self.update(
            repertoire, emitter_state, random_key
        )

        return (repertoire, emitter_state, random_key), unused
