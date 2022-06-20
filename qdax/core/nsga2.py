from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from qdax.core.containers.population_repertoire import PopulationRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Fitness, Genotype, RNGKey
from qdax.utils.pareto_front import compute_masked_pareto_front


class NSGA2:
    """Implements main functions of the NSGA2 algorithm.


    TODO: add link to paper.
    """

    def __init__(
        self,
        scoring_function: Callable[[Genotype], Fitness],
        emitter: Emitter,
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter

    @partial(jax.jit, static_argnames=("self", "population_size"))
    def init(
        self, init_genotypes: Genotype, population_size: int, random_key: RNGKey
    ) -> Tuple[NSGA2Repertoire, Optional[EmitterState], RNGKey]:

        # score initial population
        fitnesses = self._scoring_function(init_genotypes)

        # init the repertoire
        repertoire = NSGA2Repertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            population_size=population_size,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key
        )

        return repertoire, emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: NSGA2Repertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[NSGA2Repertoire, Optional[EmitterState], RNGKey]:
        """
        Do one iteration of NSGA2
        """
        # generate offsprings
        genotypes, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        # score the offsprings
        fitnesses = self._scoring_function(genotypes)

        # update the repertoire
        repertoire = repertoire.add(genotypes, fitnesses)

        return repertoire, emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def scan_update(self, carry, unused):
        # iterate over grid
        repertoire, emitter_state, random_key = carry
        repertoire, emitter_state, random_key = self.update(
            repertoire, emitter_state, random_key
        )

        # get metrics
        return (repertoire, emitter_state, random_key), unused


class NSGA2Repertoire(PopulationRepertoire):
    @jax.jit
    def _compute_crowding_distances(self, scores: Fitness, mask: jnp.ndarray):
        """
        Compute crowding distances
        """
        # Retrieve only non masked solutions
        num_solutions = scores.shape[0]
        num_objective = scores.shape[1]
        if num_solutions <= 2:
            return jnp.array([np.inf] * num_solutions)

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
        # All the candidates
        candidates = jnp.concatenate((self.genotypes, batch_of_genotypes))
        candidate_scores = jnp.concatenate((self.fitnesses, batch_of_fitnesses))

        # Track front
        to_keep_index = jnp.zeros(candidates.shape[0], dtype=np.bool)

        def compute_current_front(val):
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

        def stop_loop(val):

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

        def add_to_front(val):
            front_index, num = val
            front_index = front_index.at[highest_dist[-num]].set(True)
            num = num + 1
            val = front_index, num
            return val

        def stop_loop(val):
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
