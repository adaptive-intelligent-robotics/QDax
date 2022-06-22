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
        self, fitnesses: Fitness, mask: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute crowding distances.

        Used in the addition function.

        Args:
            fitnesses: fitnesses of the considered individuals. Here,
                fitness are vectors as we are doing multi-objective
                optimization.
            mask: a vector to mask values.

        Returns:
            The crowding distances.
        """
        # Retrieve only non masked solutions
        num_solutions = fitnesses.shape[0]
        num_objective = fitnesses.shape[1]
        if num_solutions <= 2:
            return jnp.array([jnp.inf] * num_solutions)

        else:
            # Sort solutions on each objective
            mask_dist = jnp.column_stack([mask] * fitnesses.shape[1])
            score_amplitude = jnp.max(fitnesses, axis=0) - jnp.min(fitnesses, axis=0)
            dist_fitnesses = (
                fitnesses + 3 * score_amplitude * jnp.ones_like(fitnesses) * mask_dist
            )
            sorted_index = jnp.argsort(dist_fitnesses, axis=0)
            srt_fitnesses = fitnesses[sorted_index, jnp.arange(num_objective)]
            dists = jnp.row_stack(
                [srt_fitnesses, jnp.full(num_objective, jnp.inf)]
            ) - jnp.row_stack([jnp.full(num_objective, -jnp.inf), srt_fitnesses])

            # Calculate the norm for each objective - set to NaN if all values are equal
            norm = jnp.max(srt_fitnesses, axis=0) - jnp.min(srt_fitnesses, axis=0)

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

        The population is sorted in successive pareto front. The first one
        is the global pareto front. The second one is the pareto front of the
        population where the first pareto front has been removed, etc...

        The successive pareto fronts are kept until the moment where adding a
        full pareto front would exceed the population size.

        To decide the survival of this pareto front, a crowding distance is
        computed in order to keep individuals that are spread in this last pareto
        front. Hence, the individuals with the biggest crowding distances are
        added until the population size is reached.

        Args:
            batch_of_genotypes: new genotypes that we try to add.
            batch_of_fitnesses: fitness of those new genotypes.

        Returns:
            The updated repertoire.
        """
        # All the candidates
        candidates = jax.tree_map(
            lambda x, y: jnp.concatenate((x, y), axis=0),
            self.genotypes,
            batch_of_genotypes,
        )

        candidate_fitnesses = jnp.concatenate((self.fitnesses, batch_of_fitnesses))

        first_leaf = jax.tree_leaves(candidates)[0]
        num_candidates = first_leaf.shape[0]

        # Track front
        to_keep_index = jnp.zeros(num_candidates, dtype=np.bool)

        # TODO: check shape, print etc... to be sure everything's fine

        def compute_current_front(
            val: Tuple[jnp.ndarray, jnp.ndarray]
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Body function for the while loop.
            val is a tuple with
            (current_num_solutions, to_keep_index)

            Args:
                val: Value passed through the while loop. Here, it is
                    a tuple containing two values. The current number of
                    solutions and the indexes of the front.

            Returns:
                The updated values to pass through the while loop. Updated
                number of solutions and updated front indexes.
            """
            to_keep_index, _ = val
            front_index = compute_masked_pareto_front(
                candidate_fitnesses, to_keep_index
            )

            # Add new index
            to_keep_index = to_keep_index + front_index

            # Update front & number of solutions
            return to_keep_index, front_index

        def condition_fn_1(val: Tuple[jnp.ndarray, jnp.ndarray]) -> bool:
            """_summary_

            Args:
                val: _description_

            Returns:
                _description_
            """
            to_keep_index, _ = val
            return sum(to_keep_index) < self.size  # type: ignore

        to_keep_index, front_index = jax.lax.while_loop(
            condition_fn_1,
            compute_current_front,
            (
                jnp.zeros(num_candidates, dtype=np.bool),
                jnp.zeros(num_candidates, dtype=np.bool),
            ),
        )

        # Remove Last One
        new_index = jnp.arange(start=1, stop=len(to_keep_index) + 1) * to_keep_index
        new_index = new_index * (~front_index)
        to_keep_index = new_index > 0

        # Compute crowding distances
        crowding_distances = self._compute_crowding_distances(
            candidate_fitnesses, ~front_index
        )
        crowding_distances = crowding_distances * (front_index)
        highest_dist = jnp.argsort(crowding_distances)

        def add_to_front(val: Tuple[jnp.ndarray, float]) -> Tuple[jnp.ndarray, Any]:
            front_index, num = val
            front_index = front_index.at[highest_dist[-num]].set(True)
            num = num + 1
            val = front_index, num
            return val

        def condition_fn_2(val: Tuple[jnp.ndarray, jnp.ndarray]) -> bool:
            front_index, _ = val
            return sum(to_keep_index + front_index) < self.size  # type: ignore

        # Remove the highest distances
        front_index, num = jax.lax.while_loop(
            condition_fn_2,
            add_to_front,
            (jnp.zeros(num_candidates, dtype=np.bool), 0),
        )

        # Update index
        to_keep_index = to_keep_index + front_index

        # Update (cannot use to_keep_index directly as it is dynamic)
        indices = jnp.arange(start=0, stop=num_candidates) * to_keep_index
        indices = indices + ~to_keep_index * (num_candidates)
        indices = jnp.sort(indices)[: self.size]

        new_candidates = jax.tree_map(lambda x: x[indices], candidates)
        new_scores = candidate_fitnesses[indices]

        return NSGA2Repertoire(genotypes=new_candidates, fitnesses=new_scores)
