from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax

from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import ExtraScores, Fitness, Genotype, Metrics, RNGKey


class GeneticAlgorithm:
    """"""

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[GARepertoire], Metrics],
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function

    @partial(jax.jit, static_argnames=("self", "population_size"))
    def init(
        self, init_genotypes: Genotype, population_size: int, random_key: RNGKey
    ) -> Tuple[GARepertoire, Optional[EmitterState], RNGKey]:

        # score initial genotypes
        fitnesses, extra_scores, random_key = self._scoring_function(
            init_genotypes, random_key
        )

        # init the repertoire
        repertoire = GARepertoire.init(
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
            descriptors=None,
            extra_scores=extra_scores,
        )

        return repertoire, emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: GARepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[GARepertoire, Optional[EmitterState], Metrics, RNGKey]:

        # generate offsprings
        genotypes, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        # score the offsprings
        fitnesses, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        # update the repertoire
        repertoire = repertoire.add(genotypes, fitnesses)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=None,
            extra_scores=extra_scores,
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics, random_key

    @partial(jax.jit, static_argnames=("self",))
    def scan_update(
        self,
        carry: Tuple[GARepertoire, Optional[EmitterState], RNGKey],
        unused: Any,
    ) -> Tuple[Tuple[GARepertoire, Optional[EmitterState], RNGKey], Metrics]:
        # iterate over grid
        repertoire, emitter_state, random_key = carry
        repertoire, emitter_state, metrics, random_key = self.update(
            repertoire, emitter_state, random_key
        )

        return (repertoire, emitter_state, random_key), metrics
