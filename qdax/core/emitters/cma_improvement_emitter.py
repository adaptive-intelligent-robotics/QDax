from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.cma_emitter import CMAEmitter, CMAEmitterState
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype


class CMAImprovementEmitter(CMAEmitter):
    """Class for the emitter of CMA ME from "Covariance Matrix Adaptation
    for the Rapid Illumination of Behavior Space" by Fontaine et al.

    This class implements the improvement emitter, where the update of the
    distribution is biased towards solution that improve the QD score.

    Args:
        batch_size: number of solutions sampled at each iteration
        genotype_dim: dimension of the genotype space.
        centroids: centroids used for the repertoire.
        sigma_g: standard deviation for the coefficients - called step size.
        min_count: minimum number of CMAES opt step before being considered for
            reinitialisation.
        max_count: maximum number of CMAES opt step authorized.
    """

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
