from functools import partial
from typing import Callable, Tuple

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.lax import scan, while_loop
from qdax.types import Fitness, Genotype, RNGKey
from qdax.utils.evo_utils import Solutions, init_population, sample_in_population
from qdax.utils.pareto_front import compute_masked_pareto_front


def run_nsga2(
    init_genotypes: Genotype,
    random_key: RNGKey,
    scoring_function: Callable[[Genotype], Fitness],
    crossover_function: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
    mutation_function: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
    crossover_percentage: float,
    batch_size: int,
    num_iterations: int,
    population_size: int,
) -> Solutions:
    """
    Run NSGA II [1] Optimizer

    Parameters:
        init_genotypes (Genotype): initial genotypes
        random_key (RNGKey): jax random key to use
        scoring_function (Callable): scoring function, returns scores only
        crossover_function (Callable): crossover function
        mutation_function (Callable): mutation function
        crossover_percentage (float): percentage of crossover vs mutation ([0, 1])
        batch_size (int): number of new candidates tested at each iteration
        num_iterations (int): number of iterations to perform
        population_size (int): total archive size

    Returns:
        solutions (Solutions): solutions (genotypes and scores) found by NSGA II

    [1] Deb, Kalyanmoy, et al. "A fast and elitist multiobjective genetic algorithm:
        NSGA-II." IEEE transactions on evolutionary computation 6.2 (2002): 182-197.
    """
    # jit functions
    init_function = partial(
        init_population,
        scoring_function=scoring_function,
        population_size=population_size,
    )
    init_function = jit(init_function)

    iteration_function = partial(
        do_iteration_nsga2,
        scoring_function=scoring_function,
        crossover_function=crossover_function,
        mutation_function=mutation_function,
        crossover_percentage=crossover_percentage,
        batch_size=batch_size,
    )

    @jit
    def iteration_fn(carry, unused):
        # iterate over grid
        solutions, random_key = carry
        solutions, random_key = iteration_function(solutions, random_key)

        # get metrics
        return (solutions, random_key), unused

    # init algorithm
    solutions = init_function(init_genotypes)

    # run optimization loop
    (solutions, random_key), _ = scan(
        iteration_fn, (solutions, random_key), (), length=num_iterations
    )

    return solutions


def do_iteration_nsga2(
    solutions: Solutions,
    random_key: RNGKey,
    mutation_function: Callable,
    crossover_function: Callable,
    scoring_function: Callable,
    crossover_percentage: float,
    batch_size: int,
) -> Solutions:
    """
    Do one iteration of NSGA2
    """

    n_crossover = int(batch_size * crossover_percentage)
    n_mutation = batch_size - n_crossover

    # Cross-over
    if n_crossover > 0:
        x1, random_key = sample_in_population(
            solutions, random_key, np.max((n_mutation, n_crossover))
        )
        x2, random_key = sample_in_population(solutions, random_key, n_crossover)
        x_crossover, random_key = crossover_function(x1[:n_crossover], x2, random_key)

        # Mutation
        if n_mutation > 0:
            x1, random_key = sample_in_population(
                solutions, random_key, np.max((n_mutation, n_crossover))
            )
            x_mutation, random_key = mutation_function(x1[:n_mutation], random_key)
            new_genotypes = jnp.concatenate((x_mutation, x_crossover), axis=0)
        else:
            new_genotypes = x_crossover
    else:
        x1, random_key = sample_in_population(
            solutions, random_key, np.max((n_mutation, n_crossover))
        )
        x_mutation, random_key = mutation_function(x1[:n_mutation], random_key)
        new_genotypes = x_mutation

    new_scores = scoring_function(new_genotypes)

    new_solutions = Solutions(new_genotypes, new_scores)

    solutions = update_population(solutions, new_solutions)

    return solutions, random_key


def update_population(solutions: Solutions, new_solutions: Solutions) -> Solutions:
    # All the candidates
    candidates = jnp.concatenate((solutions.genotypes, new_solutions.genotypes))
    candidate_scores = jnp.concatenate((solutions.scores, new_solutions.scores))

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
        return sum(to_keep_index) < solutions.size

    to_keep_index, front_index = while_loop(
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
    crowding_distances = compute_crowding_distances(candidate_scores, ~front_index)
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
        return sum(to_keep_index + front_index) < solutions.size

    # Remove the highest distances
    front_index, num = while_loop(
        stop_loop, add_to_front, (jnp.zeros(candidates.shape[0], dtype=np.bool), 0)
    )

    # Update index
    to_keep_index = to_keep_index + front_index

    # Update (cannot use to_keep_index directly as it is dynamic)
    indices = jnp.arange(start=0, stop=len(candidates)) * to_keep_index
    indices = indices + ~to_keep_index * (len(candidates))
    indices = jnp.sort(indices)[: solutions.size]

    new_candidates = candidates[indices]
    new_scores = candidate_scores[indices]
    new_solutions = Solutions(new_candidates, new_scores)
    return new_solutions


def compute_crowding_distances(scores: Fitness, mask: jnp.ndarray):
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
        dist_scores = scores + 3 * score_amplitude * jnp.ones_like(scores) * mask_dist
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
