"""Core components of a basic genetic algorithm."""
from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax

from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import ExtraScores, Fitness, Genotype, Metrics, RNGKey


class GeneticAlgorithm:
    """Core class of a genetic algorithm.

    This class implements default methods to run a simple
    genetic algorithm with a simple repertoire.

    Args:
        scoring_function: a function that takes a batch of genotypes and compute
            their fitnesses
        emitter: an emitter is used to suggest offsprings given a repertoire. It has
            two compulsory functions. A function that takes emits a new population, and
            a function that update the internal state of the emitter
        metrics_function: a function that takes a repertoire and compute any useful
            metric to track its evolution
    """

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
        """Initialize a GARepertoire with an initial population of genotypes.

        Args:
            init_genotypes: the initial population of genotypes
            population_size: the maximal size of the repertoire
            random_key: a random key to handle stochastic operations

        Returns:
            The initial repertoire, an initial emitter state and a new random key.
        """

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
        """
        Performs one iteration of a Genetic algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.

        Args:
            repertoire: a repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """

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
        """Rewrites the update function in a way that makes it compatible with the
        jax.lax.scan primitive.

        Args:
            carry: a tuple containing the repertoire, the emitter state and a
                random key.
            unused: unused element, necessary to respect jax.lax.scan API.

        Returns:
            The updated repertoire and emitter state, with a new random key and metrics.
        """
        # iterate over grid
        repertoire, emitter_state, random_key = carry
        repertoire, emitter_state, metrics, random_key = self.update(
            repertoire, emitter_state, random_key
        )

        return (repertoire, emitter_state, random_key), metrics
