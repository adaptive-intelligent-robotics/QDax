from __future__ import annotations

from typing import Callable, Optional, Tuple

from qdax.core.containers.mome_repertoire import MOMERepertoire
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


class MOME(MAPElites):
    """Implements Multi-Objectives MAP Elites.

    Most methods in this class are inherited from MAPElites.

    The same scoring function can be passed into both MAPElites and this class.
    We have overridden __init__ such that it takes into account the specificities
    of the Multi-Objective repertoire, particularly the pareto_front_max_length.
    In particular, we create a wrapper around the provided repertoire_init function
    to properly handle the pareto_front_max_length parameter when initializing
    the MOMERepertoire.
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MOMERepertoire], Metrics],
        pareto_front_max_length: int,
        repertoire_init: Callable[
            [Genotype, Fitness, Descriptor, Centroid, int, Optional[ExtraScores]],
            MOMERepertoire,
        ] = MOMERepertoire.init,
    ) -> None:
        self.pareto_front_max_length = pareto_front_max_length

        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function

        # This is a workaround to make the repertoire_init function work with the
        # pareto_front_max_length argument.
        def _repertoire_init(
            genotypes: Genotype,
            fitnesses: Fitness,
            descriptors: Descriptor,
            centroids: Centroid,
            extra_scores: Optional[ExtraScores] = None,
        ) -> MOMERepertoire:
            return repertoire_init(
                genotypes,
                fitnesses,
                descriptors,
                centroids,
                self.pareto_front_max_length,
                extra_scores,
            )

        self._repertoire_init = _repertoire_init
