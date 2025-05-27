"""Core components of the MAP-Elites Low-Spread algorithm."""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

from qdax.core.containers.mels_repertoire import MELSRepertoire
from qdax.core.emitters.emitter import Emitter
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
from qdax.utils.sampling import multi_sample_scoring_function


class MELS(MAPElites):
    """Core elements of the MAP-Elites Low-Spread algorithm.

    Most methods in this class are inherited from MAPElites.

    The same scoring function can be passed into both MAPElites and this class.
    We have overridden __init__ such that it takes in the scoring function and
    wraps it such that every solution is evaluated `num_samples` times.
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MELSRepertoire], Metrics],
        num_samples: int,
        repertoire_init: Callable[
            [Genotype, Fitness, Descriptor, Centroid, Optional[ExtraScores]],
            MELSRepertoire,
        ] = MELSRepertoire.init,
    ) -> None:
        self._scoring_function = partial(
            multi_sample_scoring_function,
            scoring_fn=scoring_function,
            num_samples=num_samples,
        )
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._num_samples = num_samples
        self._repertoire_init = repertoire_init
