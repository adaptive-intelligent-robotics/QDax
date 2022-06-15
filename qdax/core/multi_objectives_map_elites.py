from __future__ import annotations

from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.types import Centroid, Descriptor, Fitness, Genotype, Metrics, RNGKey
from qdax.utils.mome_utils import (
    MOQDMetrics,
    compute_hypervolume,
    compute_masked_pareto_front,
)


class MOME:
    def __init__(
        self,
        scoring_function: Callable[[Genotype], Tuple[Fitness, Descriptor]],
        batch_size: int,
    ) -> None:
        self._scoring_function = scoring_function
        self._batch_size = batch_size

    @partial(jax.jit, static_argnames=("self", "pareto_front_max_length"))
    def init(
        self,
        init_genotypes: jnp.ndarray,
        centroids: Centroid,
        pareto_front_max_length: int,
    ) -> MOMERepertoire:
        """
        Initialize a MOME grid with an initial population of genotypes. Requires
        the definition of centroids that can be computed with any method such as
        CVT or Euclidean mapping.
        """

        init_fitnesses, init_descriptors = self._scoring_function(init_genotypes)

        num_criteria = init_fitnesses.shape[1]
        dim_x = init_genotypes.shape[1]
        num_descriptors = init_descriptors.shape[1]
        num_centroids = centroids.shape[0]

        grid_fitness = -jnp.inf * jnp.ones(
            shape=(num_centroids, pareto_front_max_length, num_criteria)
        )
        grid_x = jnp.zeros(shape=(num_centroids, pareto_front_max_length, dim_x))
        grid_descriptors = jnp.zeros(
            shape=(num_centroids, pareto_front_max_length, num_descriptors)
        )

        repertoire = MOMERepertoire(  # type: ignore
            genotypes=grid_x,
            fitnesses=grid_fitness,
            descriptors=grid_descriptors,
            centroids=centroids,
        )

        repertoire = repertoire.add(
            batch_of_genotypes=init_genotypes,
            batch_of_descriptors=init_descriptors,
            batch_of_fitnesses=init_fitnesses,
        )

        return repertoire

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "crossover_function",
            "mutation_function",
            "crossover_percentage",
        ),
    )
    def update(
        self,
        repertoire: MOMERepertoire,
        random_key: RNGKey,
        crossover_function: Callable[
            [Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]
        ],
        mutation_function: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        crossover_percentage: float,
    ) -> Tuple[MOMERepertoire, RNGKey]:
        """
        Performs one iteration of the MOME algorithm.
        1. A batch of genotypes is sampled in the archive and the genotypes are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the archive.
        """
        n_crossover = int(self._batch_size * crossover_percentage)
        n_mutation = self._batch_size - n_crossover

        if n_crossover > 0:
            x1, random_key = repertoire.sample(random_key, n_crossover)
            x2, random_key = repertoire.sample(random_key, n_crossover)

            x_crossover, random_key = crossover_function(x1, x2, random_key)

        if n_mutation > 0:
            x1, random_key = repertoire.sample(random_key, n_mutation)
            x_mutation, random_key = mutation_function(x1, random_key)

        if n_crossover == 0:
            genotypes = x_mutation
        elif n_mutation == 0:
            genotypes = x_crossover
        else:
            genotypes = jnp.concatenate([x_crossover, x_mutation], axis=0)

        fitnesses, descriptors = self._scoring_function(genotypes)

        repertoire = repertoire.add(genotypes, descriptors, fitnesses)

        return repertoire, random_key


def compute_moqd_metrics(
    grid: MOMERepertoire, reference_point: jnp.ndarray
) -> MOQDMetrics:
    """
    Compute the MOQD metric given a MOME grid and a reference point.
    """
    grid_empty = grid.fitnesses == -jnp.inf
    grid_empty = jnp.all(grid_empty, axis=-1)
    grid_not_empty = ~grid_empty
    grid_not_empty = jnp.any(grid_not_empty, axis=-1)
    coverage = 100 * jnp.mean(grid_not_empty)
    hypervolume_function = partial(compute_hypervolume, reference_point=reference_point)
    moqd_scores = jax.vmap(hypervolume_function)(grid.fitnesses)
    moqd_scores = jnp.where(grid_not_empty, moqd_scores, -jnp.inf)
    max_hypervolume = jnp.max(moqd_scores)
    max_scores = jnp.max(grid.fitnesses, axis=(0, 1))
    max_sum_scores = jnp.max(jnp.sum(grid.fitnesses, axis=-1), axis=(0, 1))
    num_solutions = jnp.sum(~grid_empty)
    (
        pareto_front,
        _,
    ) = compute_global_pareto_front(grid)

    global_hypervolume = compute_hypervolume(
        pareto_front, reference_point=reference_point
    )
    metrics = MOQDMetrics(
        moqd_score=moqd_scores,
        max_hypervolume=max_hypervolume,
        max_scores=max_scores,
        max_sum_scores=max_sum_scores,
        coverage=coverage,
        number_solutions=num_solutions,
        global_hypervolume=global_hypervolume,
    )

    return metrics


def compute_global_pareto_front(
    grid: MOMERepertoire,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Merge all the pareto fronts of the MOME grid into a single one called global.
    """
    scores = jnp.concatenate(grid.fitnesses, axis=0)
    mask = jnp.any(scores == -jnp.inf, axis=-1)
    pareto_bool = compute_masked_pareto_front(scores, mask)
    pareto_front = scores - jnp.inf * (~jnp.array([pareto_bool, pareto_bool]).T)

    return pareto_front, pareto_bool


def add_init_metrics(metrics: MOQDMetrics, init_metrics: MOQDMetrics) -> MOQDMetrics:
    """
    Append an initial metric to the run metrics.
    """
    metrics = jax.tree_multimap(
        lambda x, y: jnp.concatenate((jnp.expand_dims(x, axis=0), y), axis=0),
        init_metrics,
        metrics,
    )
    return metrics
