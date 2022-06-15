
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.cm as cm
from jax import vmap
from jax.lax import scan
from matplotlib.pyplot import plt
from qdax.analysis.plotting import (get_voronoi_finite_polygons_2d,
                                    vector_to_rgb)
from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.types import Centroid, Descriptor, Genotype, Metrics, RNGKey, Scores
from qdax.utils.evo_utils import MOQDMetrics, compute_hypervolume
from qdax.utils.pareto_front import compute_masked_pareto_front


def init_mome(
    init_genotypes: Genotype,
    centroids: Centroid,
    scoring_function: Callable[[Genotype], Tuple[Scores, Descriptor]],
    pareto_front_max_length: int,
) -> MOMERepertoire:
    """
    Initialize a MOME grid with an initial population of genotypes. Requires
    the definition of centroids that can be computed with any method such as
    CVT or Euclidean mapping.
    """

    init_fitnesses, init_descriptors = scoring_function(init_genotypes)

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

    map_elites_grid = MOMERepertoire(  # type: ignore
        grid=grid_x,
        grid_scores=grid_fitness,
        grid_descriptors=grid_descriptors,
        centroids=centroids,
    )

    map_elites_grid = map_elites_grid.add_in_grid(
        genotypes=init_genotypes,
        batch_of_descriptors=init_descriptors,
        batch_of_fitnesses=init_fitnesses,
    )

    return map_elites_grid


def do_mome_iteration(
    map_elites_grid: MOMERepertoire,
    random_key: RNGKey,
    scoring_function: Callable[[Genotype], Tuple[Scores, Descriptor]],
    crossover_function: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
    mutation_function: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
    crossover_percentage: float,
    batch_size: int,
) -> Tuple[MOMERepertoire, RNGKey]:
    """
    Performs one iteration of the MOME algorithm.
    1. A batch of genotypes is sampled in the archive and the genotypes are copied.
    2. The copies are mutated and crossed-over
    3. The obtained offsprings are scored and then added to the archive.
    """
    n_crossover = int(batch_size * crossover_percentage)
    n_mutation = batch_size - n_crossover

    if n_crossover > 0:
        x1, random_key = map_elites_grid.sample_in_grid(random_key, n_crossover)
        x2, random_key = map_elites_grid.sample_in_grid(random_key, n_crossover)

        x_crossover, random_key = crossover_function(x1, x2, random_key)

    if n_mutation > 0:
        x1, random_key = map_elites_grid.sample_in_grid(random_key, n_mutation)
        x_mutation, random_key = mutation_function(x1, random_key)

    if n_crossover == 0:
        genotypes = x_mutation
    elif n_mutation == 0:
        genotypes = x_crossover
    else:
        genotypes = jnp.concatenate([x_crossover, x_mutation], axis=0)

    fitnesses, descriptors = scoring_function(genotypes)

    map_elites_grid = map_elites_grid.add_in_grid(genotypes, descriptors, fitnesses)

    # Generate new key
    random_key, _ = random.split(random_key)

    return map_elites_grid, random_key


def run_mome(
    centroids: Centroid,
    init_genotypes: Genotype,
    random_key: RNGKey,
    scoring_function: Callable[[Genotype], Tuple[Scores, Descriptor]],
    crossover_function: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
    mutation_function: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
    crossover_percentage: float,
    batch_size: int,
    num_iterations: int,
    pareto_front_max_length: int,
    reference_point: jnp.ndarray,
) -> Tuple[MOMERepertoire, MOQDMetrics]:
    """
    Run the MOME algorithm. This function initialize the MOME grid and
    then performs num_iterations iterations. It returns the final grid and
    the MOQD metrics for each iteration.
    """
    # jit functions
    init_function = partial(
        init_mome,
        centroids=centroids,
        scoring_function=scoring_function,
        pareto_front_max_length=pareto_front_max_length,
    )
    init_function = jax.jit(init_function)

    iteration_function = partial(
        do_mome_iteration,
        scoring_function=scoring_function,
        crossover_function=crossover_function,
        mutation_function=mutation_function,
        crossover_percentage=crossover_percentage,
        batch_size=batch_size,
    )

    @jax.jit
    def iteration_fn(
        carry: Tuple[MOMERepertoire, jnp.ndarray], unused: Any
    ) -> Tuple[Tuple[MOMERepertoire, RNGKey], Metrics]:
        # iterate over grid
        grid, random_key = carry
        grid, random_key = iteration_function(grid, random_key)

        # get metrics
        metrics = compute_moqd_metrics(grid, reference_point)
        return (grid, random_key), metrics

    # init algorithm
    map_elites_grid = init_function(init_genotypes)
    init_metrics = compute_moqd_metrics(map_elites_grid, reference_point)

    # run optimization loop
    (map_elites_grid, random_key), metrics = scan(
        iteration_fn, (map_elites_grid, random_key), (), length=num_iterations
    )
    metrics = add_init_metrics(metrics, init_metrics)

    return map_elites_grid, metrics


def compute_moqd_metrics(
    grid: MOMERepertoire, reference_point: jnp.ndarray
) -> MOQDMetrics:
    """
    Compute the MOQD metric given a MOME grid and a reference point.
    """
    grid_empty = grid.grid_scores == -jnp.inf
    grid_empty = jnp.all(grid_empty, axis=-1)
    grid_not_empty = ~grid_empty
    grid_not_empty = jnp.any(grid_not_empty, axis=-1)
    coverage = 100 * jnp.mean(grid_not_empty)
    hypervolume_function = partial(compute_hypervolume, reference_point=reference_point)
    moqd_scores = vmap(hypervolume_function)(grid.grid_scores)
    moqd_scores = jnp.where(grid_not_empty, moqd_scores, -jnp.inf)
    max_hypervolume = jnp.max(moqd_scores)
    max_scores = jnp.max(grid.grid_scores, axis=(0, 1))
    max_sum_scores = jnp.max(jnp.sum(grid.grid_scores, axis=-1), axis=(0, 1))
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
    scores = jnp.concatenate(grid.grid_scores, axis=0)
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


def plot_mome_pareto_fronts(
    centroids: jnp.ndarray,
    map_elites_grid: MOMERepertoire,
    maxval: float,
    minval: float,
    axes: Optional[plt.Axes] = None,
    color_style: Optional[str] = "hsv",
    with_global: Optional[bool] = False,
) -> plt.Axes:
    grid_scores = map_elites_grid.grid_scores
    grid_descriptors = map_elites_grid.grid_descriptors

    assert grid_scores.shape[-1] == grid_descriptors.shape[-1] == 2
    assert color_style in ["hsv", "spectral"], "color_style must be hsv or spectral"

    num_centroids = len(centroids)
    grid_empty = jnp.any(grid_scores == -jnp.inf, axis=-1)

    # Extract polar coordinates
    if color_style == "hsv":
        center = jnp.array([(maxval - minval) / 2] * centroids.shape[1])
        polars = jnp.stack(
            (
                jnp.sqrt((jnp.sum((centroids - center) ** 2, axis=-1)))
                / (maxval - minval)
                / jnp.sqrt(2),
                jnp.arctan((centroids - center)[:, 1] / (centroids - center)[:, 0]),
            ),
            axis=-1,
        )
    elif color_style == "spectral":
        cmap = cm.get_cmap("Spectral")

    if axes is None:
        _, axes = plt.subplots(ncols=2, figsize=(12, 6))

    for i in range(num_centroids):
        if jnp.sum(~grid_empty[i]) > 0:
            cell_scores = grid_scores[i][~grid_empty[i]]
            cell = grid_descriptors[i][~grid_empty[i]]
            if color_style == "hsv":
                color = vector_to_rgb(polars[i, 1], polars[i, 0])
            else:
                color = cmap((centroids[i, 0] - minval) / (maxval - minval))
            axes[0].plot(cell_scores[:, 0], cell_scores[:, 1], "o", color=color)

            axes[1].plot(cell[:, 0], cell[:, 1], "o", color=color)

    # create the regions and vertices from centroids
    regions, vertices = get_voronoi_finite_polygons_2d(centroids)

    # fill the plot with contours
    for region in regions:
        polygon = vertices[region]
        axes[1].fill(
            *zip(*polygon), alpha=0.2, edgecolor="black", facecolor="white", lw=1
        )
    axes[0].set_title("Scores")
    axes[1].set_title("Descriptor")
    axes[1].set_xlim(minval, maxval)
    axes[1].set_ylim(minval, maxval)

    if with_global:
        global_pareto_front, pareto_bool = compute_global_pareto_front(map_elites_grid)
        global_pareto_descriptors = jnp.concatenate(grid_descriptors)[pareto_bool]
        axes[0].scatter(
            global_pareto_front[:, 0],
            global_pareto_front[:, 1],
            marker="o",
            edgecolors="black",
            facecolors="none",
            zorder=3,
            label="Global Pareto Front",
        )
        sorted_index = jnp.argsort(global_pareto_front[:, 0])
        axes[0].plot(
            global_pareto_front[sorted_index, 0],
            global_pareto_front[sorted_index, 1],
            linestyle="--",
            linewidth=2,
            color="k",
            zorder=3,
        )
        axes[1].scatter(
            global_pareto_descriptors[:, 0],
            global_pareto_descriptors[:, 1],
            marker="o",
            edgecolors="black",
            facecolors="none",
            zorder=3,
            label="Global Pareto Descriptor",
        )

    return axes
