"""Core components of the Dominated Novelty Search algorithm."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import jax

from qdax.core.containers.dns_repertoire import DominatedNoveltyRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import (
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)


class DominatedNoveltySearch:
    """Core elements of the Dominated Novelty Search (DNS) algorithm.

    DNS maintains a flat population without tessellation and selects survivors
    using dominated novelty computed in descriptor space.

    Args:
        scoring_function: a function that takes a batch of genotypes and computes
            their fitnesses, descriptors and optional extra scores.
        emitter: an emitter used to propose offspring and update its internal state.
        metrics_function: a function that takes a DNS repertoire and computes
            metrics to track the evolution.
        population_size: maximum number of individuals maintained.
        k: number of nearest neighbors used to compute novelty/dominated novelty.
        repertoire_init: optional custom initialization function for the repertoire.
    """

    def __init__(
        self,
        scoring_function: Optional[
            Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores]]
        ],
        emitter: Emitter,
        metrics_function: Callable[[DominatedNoveltyRepertoire], Metrics],
        population_size: int,
        k: int,
        repertoire_init: Callable[
            [Genotype, Fitness, Descriptor, int, int, Optional[ExtraScores]],
            DominatedNoveltyRepertoire,
        ] = DominatedNoveltyRepertoire.init,
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._population_size = population_size
        self._k = k

        # Wrapper to bind population_size and k
        def _repertoire_init(
            genotypes: Genotype,
            fitnesses: Fitness,
            descriptors: Descriptor,
            _population_size: int,
            _k: int,
            extra_scores: Optional[ExtraScores] = None,
        ) -> DominatedNoveltyRepertoire:
            del _population_size, _k  # provided via closure below
            return repertoire_init(
                genotypes,
                fitnesses,
                descriptors,
                self._population_size,
                self._k,
                extra_scores,
            )

        self._repertoire_init = _repertoire_init

    def init(
        self,
        genotypes: Genotype,
        key: RNGKey,
    ) -> Tuple[DominatedNoveltyRepertoire, Optional[EmitterState], Metrics]:
        """
        Initialize a DNS repertoire with an initial population of genotypes.

        Args:
            genotypes: initial genotypes, pytree (batch_size, ...)
            key: a random key used for stochastic operations.

        Returns:
            An initialized DNS repertoire with the initial state of the emitter
            and the initial metrics.
        """
        if self._scoring_function is None:
            raise ValueError("Scoring function is not set.")

        # score initial genotypes
        key, subkey = jax.random.split(key)
        (fitnesses, descriptors, extra_scores) = self._scoring_function(
            genotypes, subkey
        )

        repertoire, emitter_state, metrics = self.init_ask_tell(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            key=key,
            extra_scores=extra_scores,
        )
        return repertoire, emitter_state, metrics

    def init_ask_tell(
        self,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        key: RNGKey,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Tuple[DominatedNoveltyRepertoire, Optional[EmitterState], Metrics]:
        """Initialize a DNS repertoire with evaluated initial genotypes."""
        if extra_scores is None:
            extra_scores = {}

        # init the repertoire
        repertoire = self._repertoire_init(
            genotypes,
            fitnesses,
            descriptors,
            self._population_size,
            self._k,
            extra_scores,
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

        # calculate the initial metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics

    def update(
        self,
        repertoire: DominatedNoveltyRepertoire,
        emitter_state: Optional[EmitterState],
        key: RNGKey,
    ) -> Tuple[DominatedNoveltyRepertoire, Optional[EmitterState], Metrics]:
        """
        Performs one iteration of DNS:
        1. Ask the emitter for offsprings based on the current repertoire.
        2. Score offsprings to obtain fitnesses and descriptors.
        3. Add them to the repertoire using dominated novelty selection.
        4. Update the emitter state and compute metrics.
        """
        if self._scoring_function is None:
            raise ValueError("Scoring function is not set.")

        # generate offsprings with the emitter
        key, subkey = jax.random.split(key)
        genotypes, extra_info = self.ask(repertoire, emitter_state, subkey)

        # score the offsprings
        key, subkey = jax.random.split(key)
        (fitnesses, descriptors, extra_scores) = self._scoring_function(
            genotypes, subkey
        )

        repertoire, emitter_state, metrics = self.tell(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            repertoire=repertoire,
            emitter_state=emitter_state,
            extra_scores=extra_scores,
            extra_info=extra_info,
        )
        return repertoire, emitter_state, metrics

    def scan_update(
        self,
        carry: Tuple[DominatedNoveltyRepertoire, Optional[EmitterState], RNGKey],
        _: Any,
    ) -> Tuple[
        Tuple[DominatedNoveltyRepertoire, Optional[EmitterState], RNGKey], Metrics
    ]:
        """scan-compatible wrapper around update."""
        repertoire, emitter_state, key = carry
        key, subkey = jax.random.split(key)
        (
            repertoire,
            emitter_state,
            metrics,
        ) = self.update(
            repertoire,
            emitter_state,
            subkey,
        )

        return (repertoire, emitter_state, key), metrics

    def ask(
        self,
        repertoire: DominatedNoveltyRepertoire,
        emitter_state: Optional[EmitterState],
        key: RNGKey,
    ) -> Tuple[Genotype, ExtraScores]:
        """Ask the emitter to generate a new batch of genotypes."""
        key, subkey = jax.random.split(key)
        genotypes, extra_info = self._emitter.emit(repertoire, emitter_state, subkey)
        return genotypes, extra_info

    def tell(
        self,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        repertoire: DominatedNoveltyRepertoire,
        emitter_state: Optional[EmitterState],
        extra_scores: Optional[ExtraScores] = None,
        extra_info: Optional[ExtraScores] = None,
    ) -> Tuple[DominatedNoveltyRepertoire, Optional[EmitterState], Metrics]:
        """Add new genotypes to the repertoire and update the emitter state."""
        if extra_scores is None:
            extra_scores = {}
        if extra_info is None:
            extra_info = {}

        # add genotypes in the repertoire
        repertoire = repertoire.add(genotypes, descriptors, fitnesses, extra_scores)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores={**extra_scores, **extra_info},
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics
