"""Core components of a basic genetic algorithm."""

from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax

from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import ExtraScores, Fitness, Genotype, Metrics, RNGKey


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
        scoring_function: Callable[[Genotype, RNGKey], Tuple[Fitness, ExtraScores]],
        emitter: Emitter,
        metrics_function: Callable[[GARepertoire], Metrics],
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function

    @partial(jax.jit, static_argnames=("self", "population_size"))
    def init(
        self, genotypes: Genotype, population_size: int, key: RNGKey
    ) -> Tuple[GARepertoire, Optional[EmitterState], Metrics]:
        """Initialize a GARepertoire with an initial population of genotypes.

        Args:
            genotypes: the initial population of genotypes
            population_size: the maximal size of the repertoire
            key: a random key to handle stochastic operations

        Returns:
            The initial repertoire, an initial emitter state and a new random key.
        """

        # score initial genotypes
        key, subkey = jax.random.split(key)
        fitnesses, extra_scores = self._scoring_function(genotypes, subkey)

        # init the repertoire
        repertoire = GARepertoire.init(
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

        # calculate the initial metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: GARepertoire,
        emitter_state: Optional[EmitterState],
        key: RNGKey,
    ) -> Tuple[GARepertoire, Optional[EmitterState], Metrics]:
        """
        Performs one iteration of a Genetic algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.

        Args:
            repertoire: a repertoire
            emitter_state: state of the emitter
            key: a jax PRNG random key

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """

        # generate offsprings
        key, subkey = jax.random.split(key)
        genotypes, extra_info = self._emitter.emit(repertoire, emitter_state, subkey)

        # score the offsprings
        key, subkey = jax.random.split(key)
        fitnesses, extra_scores = self._scoring_function(genotypes, subkey)

        # update the repertoire
        repertoire = repertoire.add(genotypes, fitnesses)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=None,
            extra_scores={**extra_scores, **extra_info},
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics

    @partial(jax.jit, static_argnames=("self",))
    def scan_update(
        self,
        carry: Tuple[GARepertoire, Optional[EmitterState], RNGKey],
        _: Any,
    ) -> Tuple[Tuple[GARepertoire, Optional[EmitterState], RNGKey], Metrics]:
        """Rewrites the update function in a way that makes it compatible with the
        jax.lax.scan primitive.

        Args:
            carry: a tuple containing the repertoire, the emitter state and a
                random key.
            _: unused element, necessary to respect jax.lax.scan API.

        Returns:
            The updated repertoire and emitter state, with a new random key and metrics.
        """
        # iterate over grid
        repertoire, emitter_state, key = carry
        key, subkey = jax.random.split(key)
        repertoire, emitter_state, metrics = self.update(
            repertoire, emitter_state, subkey
        )

        return (repertoire, emitter_state, key), metrics
