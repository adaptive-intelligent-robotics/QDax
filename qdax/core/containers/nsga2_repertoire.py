from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.types import Fitness, Genotype
from qdax.utils.pareto_front import compute_masked_pareto_front


class NSGA2Repertoire(GARepertoire):
    """Repertoire used for the NSGA2 algorithm.

    Inherits from the GARepertoire. The data stored are the genotypes
    and there fitness. Several functions are inherited from GARepertoire,
    including size, save, sample and init.
    """

    @jax.jit
    def _compute_crowding_distances(
        self, scores: Fitness, mask: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute crowding distances
        """
        # Retrieve only non masked solutions
        num_solutions = scores.shape[0]
        num_objective = scores.shape[1]
        if num_solutions <= 2:
            return jnp.array([jnp.inf] * num_solutions)

        else:
            # Sort solutions on each objective
            mask_dist = jnp.column_stack([mask] * scores.shape[1])
            score_amplitude = jnp.max(scores, axis=0) - jnp.min(scores, axis=0)
            dist_scores = (
                scores + 3 * score_amplitude * jnp.ones_like(scores) * mask_dist
            )
            sorted_index = jnp.argsort(dist_scores, axis=0)
            srt_scores = scores[sorted_index, jnp.arange(num_objective)]
            dists = jnp.row_stack(
                [srt_scores, jnp.full(num_objective, jnp.inf)]
            ) - jnp.row_stack([jnp.full(num_objective, -jnp.inf), srt_scores])

            # Calculate the norm for each objective - set to NaN if all values are equal
            norm = jnp.max(srt_scores, axis=0) - jnp.min(srt_scores, axis=0)

            # Prepare the distance to last and next vectors
            dist_to_last, dist_to_next = dists, dists
            dist_to_last = dists[:-1] / norm
            dist_to_next = dists[1:] / norm

            # Sum up the distances and reorder
            j = jnp.argsort(sorted_index, axis=0)
            crowding_distances = (
                jnp.sum(
                    (
                        dist_to_last[j, jnp.arange(num_objective)]
                        + dist_to_next[j, jnp.arange(num_objective)]
                    ),
                    axis=1,
                )
                / num_objective
            )

            return crowding_distances

    @jax.jit
    def add(
        self, batch_of_genotypes: Genotype, batch_of_fitnesses: Fitness
    ) -> NSGA2Repertoire:
        """Implements the repertoire addition rules.

        TODO: add explanation of the addition rule.

        Args:
            batch_of_genotypes: new genotypes that we try to add.
            batch_of_fitnesses: fitness of those new genotypes.

        Returns:
            The updated repertoire.
        """
        # All the candidates
        candidates = jnp.concatenate((self.genotypes, batch_of_genotypes))
        candidate_scores = jnp.concatenate((self.fitnesses, batch_of_fitnesses))

        # Track front
        to_keep_index = jnp.zeros(candidates.shape[0], dtype=np.bool)

        def compute_current_front(
            val: Tuple[jnp.ndarray, jnp.ndarray]
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            Body function for the loop, val is a tuple with
            (current_num_solutions, to_keep_index)
            """
            to_keep_index, _ = val
            front_index = compute_masked_pareto_front(candidate_scores, to_keep_index)

            # Add new index
            to_keep_index = to_keep_index + front_index

            # Update front & number of solutions
            return to_keep_index, front_index

        def stop_loop(val: Tuple[jnp.ndarray, jnp.ndarray]) -> bool:

            """
            Stop function for the loop, val is a tuple with
            (current_num_solutions, to_keep_index)
            """
            to_keep_index, _ = val
            return sum(to_keep_index) < self.size

        to_keep_index, front_index = jax.lax.while_loop(
            stop_loop,
            compute_current_front,
            (
                jnp.zeros(candidates.shape[0], dtype=np.bool),
                jnp.zeros(candidates.shape[0], dtype=np.bool),
            ),
        )

        # Remove Last One
        new_index = jnp.arange(start=1, stop=len(to_keep_index) + 1) * to_keep_index
        new_index = new_index * (~front_index)
        to_keep_index = new_index > 0

        # Compute crowding distances
        crowding_distances = self._compute_crowding_distances(
            candidate_scores, ~front_index
        )
        crowding_distances = crowding_distances * (front_index)
        highest_dist = jnp.argsort(crowding_distances)

        def add_to_front(val: Tuple[jnp.ndarray, float]) -> Tuple[jnp.ndarray, Any]:
            front_index, num = val
            front_index = front_index.at[highest_dist[-num]].set(True)
            num = num + 1
            val = front_index, num
            return val

        def stop_loop(val: Tuple[jnp.ndarray, jnp.ndarray]) -> bool:
            front_index, _ = val
            return sum(to_keep_index + front_index) < self.size

        # Remove the highest distances
        front_index, num = jax.lax.while_loop(
            stop_loop, add_to_front, (jnp.zeros(candidates.shape[0], dtype=np.bool), 0)
        )

        # Update index
        to_keep_index = to_keep_index + front_index

        # Update (cannot use to_keep_index directly as it is dynamic)
        indices = jnp.arange(start=0, stop=len(candidates)) * to_keep_index
        indices = indices + ~to_keep_index * (len(candidates))
        indices = jnp.sort(indices)[: self.size]

        new_candidates = candidates[indices]
        new_scores = candidate_scores[indices]

        return NSGA2Repertoire(genotypes=new_candidates, fitnesses=new_scores)
