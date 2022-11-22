from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.cmaes import CMAES, CMAESState
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax.core.emitters.cma_emitter import CMAEmitter, CMAEmitterState
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class CMAImprovementEmitter(CMAEmitter):
    def _ranking_criteria(
        self,
        emitter_state: CMAEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
        improvements: jnp.ndarray,
    ) -> jnp.ndarray:
        """Defines how the genotypes should be sorted. Impacts the update
        of the CMAES state. In the end, this defines the type of CMAES emitter
        used (optimizing, random direction or improvement).

        Args:
            emitter_state: current state of the emitter.
            repertoire: latest repertoire of genotypes.
            genotypes: emitted genotypes.
            fitnesses: corresponding fitnesses.
            descriptors: corresponding fitnesses.
            extra_scores: corresponding extra scores.
            improvements: improvments of the emitted genotypes. This corresponds
                to the difference between their fitness and the fitness of the
                individual occupying the cell of corresponding fitness.

        Returns:
            The values to take into account in order to rank the emitted genotypes.
            Here, it's the improvement, or the fitness when the cell was previously
            unoccupied. Additionally, genotypes that discovered a new cell are
            given on offset to be ranked in front of other genotypes.
        """

        # condition for being a new cell
        condition = improvements == jnp.inf

        # criteria: fitness if new cell, improvement else
        ranking_criteria = jnp.where(condition, x=fitnesses, y=improvements)

        # make sure to have all the new cells first
        new_cell_offset = jnp.max(ranking_criteria) - jnp.min(ranking_criteria)

        ranking_criteria = jnp.where(
            condition, x=ranking_criteria + new_cell_offset, y=ranking_criteria
        )

        return ranking_criteria  # type: ignore
