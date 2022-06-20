from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from qdax.types import Fitness, Genotype, RNGKey
from qdax.utils.evo_utils import Solutions, init_population, sample_in_population


def do_iteration_spea2(
    solutions: Solutions,
    random_key: RNGKey,
    mutation_function: Callable[[Genotype], Genotype],
    crossover_function: Callable[[Genotype, Genotype], Genotype],
    scoring_function: Callable[[Genotype], Fitness],
    crossover_percentage: float,
    batch_size: int,
    num_neighbours: int,
) -> Solutions:
    """
    Do one iteration of SPEA2
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

    solutions = update_population(solutions, new_solutions, num_neighbours)

    return solutions, random_key


def compute_strength_scores(
    solutions: Solutions, new_solutions: Solutions, num_neighbours: int
) -> jnp.ndarray:
    """
    Compute the strength scores (defined for a solution by the number of solutions
    dominating it)
    """
    scores = jnp.concatenate((solutions.scores, new_solutions.scores), axis=0)
    dominates = jnp.all((scores - jnp.expand_dims(scores, axis=1)) > 0, axis=-1)
    strength_scores = jnp.sum(dominates, axis=1)
    distance_matrix = jnp.sum((scores - jnp.expand_dims(scores, axis=1)) ** 2, axis=-1)
    densities = jnp.sum(
        jnp.sort(distance_matrix, axis=1)[:, : num_neighbours + 1], axis=1
    )

    strength_scores = strength_scores + 1 / (1 + densities)
    strength_scores = jnp.nan_to_num(strength_scores, nan=solutions.size + 2)

    return strength_scores


def update_population(
    solutions: Solutions, new_solutions: Solutions, num_neighbours: int
) -> Solutions:
    """
    Updates the population with the new solutions
    """
    # All the candidates
    candidates = jnp.concatenate((solutions.genotypes, new_solutions.genotypes))
    candidate_scores = jnp.concatenate((solutions.scores, new_solutions.scores))

    # Track front
    strength_scores = compute_strength_scores(solutions, new_solutions, num_neighbours)
    indices = jnp.argsort(strength_scores)[: len(solutions.genotypes)]
    new_candidates = candidates[indices]
    new_scores = candidate_scores[indices]
    new_solutions = Solutions(new_candidates, new_scores)
    return new_solutions


def run_spea2(
    init_genotypes: Genotype,
    random_key: RNGKey,
    scoring_function: Callable[[Genotype], Fitness],
    crossover_function: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
    mutation_function: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
    crossover_percentage: float,
    batch_size: int,
    num_iterations: int,
    num_neighbours: int,
    population_size: int,
) -> Solutions:
    """
    Run SPEA2 [1] Optimizer

    Parameters:
        init_genotypes (Genotype): initial genotypes
        random_key (RNGKey): jax random key to use
        scoring_function (Callable): scoring function, returns scores only
        crossover_function (Callable): crossover function
        mutation_function (Callable): mutation function
        crossover_percentage (float): percentage of crossover vs mutation ([0, 1])
        batch_size (int): number of new candidates tested at each iteration
        num_iterations (int): number of iterations to perform
        num_neighbours (int): number of neighbours in the density estimation
        population_size (int): total archive size

    Returns:
        solutions (Solutions): solutions (genotypes and scores) found by SPEA2

    [1] Zitzler, Eckart, Marco Laumanns, and Lothar Thiele. "SPEA2: Improving the
        strength Pareto evolutionary algorithm." TIK-report 103 (2001).
    """
    # jit functions
    init_function = partial(
        init_population,
        scoring_function=scoring_function,
        population_size=population_size,
    )
    init_function = jax.jit(init_function)

    iteration_function = partial(
        do_iteration_spea2,
        scoring_function=scoring_function,
        crossover_function=crossover_function,
        mutation_function=mutation_function,
        crossover_percentage=crossover_percentage,
        batch_size=batch_size,
        num_neighbours=num_neighbours,
    )

    # init algorithm
    solutions = init_function(init_genotypes)

    @jax.jit
    def iteration_fn(carry, unused):
        # iterate over grid
        solutions, random_key = carry
        solutions, random_key = iteration_function(solutions, random_key)

        return (solutions, random_key), unused

    # init algorithm
    solutions = init_function(init_genotypes)

    # run optimization loop
    (solutions, random_key), _ = jax.lax.scan(
        iteration_fn, (solutions, random_key), (), length=num_iterations
    )

    return solutions
