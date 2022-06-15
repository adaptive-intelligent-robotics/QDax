from typing import Any, Optional

import jax
import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from qdax.utils.plotting import get_voronoi_finite_polygons_2d
from typing_extensions import TypeAlias


def compute_pareto_dominance(criteria_point, batch_of_criteria):
    """
    Returns if a point is pareto dominated given a set of points or not.
    We use maximization convention here.
    """
    # criteria_point of shape (num_criteria,)
    # batch_of_criteria of shape (num_points, num_criteria)
    diff = jnp.subtract(batch_of_criteria, criteria_point)
    return jnp.any(jnp.all(diff > 0, axis=-1))


def compute_pareto_front(batch_of_criteria):
    """
    Returns an array of boolean that states for each element if it is in
    the pareto front or not.
    """
    # batch_of_criteria of shape (num_points, num_criteria)
    func = jax.vmap(lambda x: ~compute_pareto_dominance(x, batch_of_criteria))
    return func(batch_of_criteria)


def compute_masked_pareto_dominance(criteria_point, batch_of_criteria, mask):
    """
    Returns if a point is pareto dominated given a set of points or not.
    We use maximization convention here.
    This function is to be used with constant size batches of criteria,
    thus a mask is used to know which values are padded.
    """
    # criteria_point of shape (num_criteria,)
    # batch_of_criteria of shape (batch_size, num_criteria)
    # mask of shape (batch_size,), 1.0 where there is not element, 0 otherwise
    diff = jnp.subtract(batch_of_criteria, criteria_point)
    neutral_values = -jnp.ones_like(diff)
    diff = jax.vmap(lambda x1, x2: jnp.where(mask, x1, x2), (1, 1), 1)(
        neutral_values, diff
    )
    return jnp.any(jnp.all(diff > 0, axis=-1))


def compute_masked_pareto_front(batch_of_criteria, mask):
    """
    Returns an array of boolean that states for each element if it is to be
    considered or not. This function works is to be used constant size batches of
    criteria, thus a mask is used to know which values are padded.
    """
    func = jax.vmap(
        lambda x: ~compute_masked_pareto_dominance(x, batch_of_criteria, mask)
    )
    return func(batch_of_criteria) * ~mask


def sample_in_masked_pareto_front(
    pareto_front_x, mask, num_samples: int, random_key: jnp.ndarray
):
    """
    Sample num_samples elements in masked pareto front.
    """
    random_key, sub_key = jax.random.split(random_key)
    p = (1.0 - mask) / jnp.sum(mask)
    return (
        jax.random.choice(sub_key, pareto_front_x, shape=(num_samples,), p=p),
        random_key,
    )


def update_masked_pareto_front(
    pareto_front_fitness,
    pareto_front_x,
    mask,
    new_batch_of_criteria,
    new_batch_of_x,
    new_mask,
):
    """
    Takes a fixed size pareto front, its mask and new points to add.
    Returns updated front and mask.
    """
    batch_size = new_batch_of_criteria.shape[0]
    pareto_front_len = pareto_front_fitness.shape[0]
    num_criteria = new_batch_of_criteria.shape[1]
    x_dim = new_batch_of_x.shape[1]

    cat_mask = jnp.concatenate([mask, new_mask], axis=-1)
    cat_f = jnp.concatenate([pareto_front_fitness, new_batch_of_criteria], axis=0)
    cat_x = jnp.concatenate([pareto_front_x, new_batch_of_x], axis=0)
    cat_bool_front = compute_masked_pareto_front(batch_of_criteria=cat_f, mask=cat_mask)
    indices = jnp.arange(start=0, stop=pareto_front_len + batch_size) * cat_bool_front
    indices = indices + ~cat_bool_front * (batch_size + pareto_front_len - 1)
    indices = jnp.sort(indices)
    new_front_fitness = jnp.take(cat_f, indices, axis=0)
    new_front_x = jnp.take(cat_x, indices, axis=0)

    num_front_elements = jnp.sum(cat_bool_front)
    new_mask_indices = jnp.arange(start=0, stop=batch_size + pareto_front_len)
    new_mask_indices = (num_front_elements - new_mask_indices) > 0

    new_mask = jnp.where(
        new_mask_indices,
        jnp.ones(shape=batch_size + pareto_front_len, dtype=bool),
        jnp.zeros(shape=batch_size + pareto_front_len, dtype=bool),
    )

    fitness_mask = jnp.repeat(jnp.expand_dims(new_mask, axis=-1), num_criteria, axis=-1)
    new_front_fitness = new_front_fitness * fitness_mask
    new_front_fitness = new_front_fitness[: len(pareto_front_fitness), :]

    x_mask = jnp.repeat(jnp.expand_dims(new_mask, axis=-1), x_dim, axis=-1)
    new_front_x = new_front_x * x_mask
    new_front_x = new_front_x[: len(pareto_front_fitness), :]

    new_mask = ~new_mask[: len(pareto_front_fitness)]

    return new_front_fitness, new_front_x, new_mask


def vector_to_rgb(angle: float, absolute: float) -> Any:
    """
    Returns a color based on polar coordinates.
    """

    # normalize angle
    angle = angle % (2 * np.pi)
    if angle < 0:
        angle += 2 * np.pi

    # rise absolute to avoid black colours
    absolute = (absolute + 0.5) / 1.5

    return mpl.colors.hsv_to_rgb((angle / 2 / np.pi, 1, absolute))


# Define Metrics
MOQDScore: TypeAlias = jnp.ndarray
MaxHypervolume: TypeAlias = jnp.ndarray
MaxScores: TypeAlias = jnp.ndarray
MaxSumScores: TypeAlias = jnp.ndarray
Coverage: TypeAlias = jnp.ndarray
NumSolutions: TypeAlias = jnp.ndarray
GlobalHypervolume: TypeAlias = jnp.ndarray


class MOQDMetrics(flax.struct.PyTreeNode):
    """
    Class to store Multi-Objective QD performance metrics.

        moqd_score: Hypervolume of the Pareto Front in each cell (n_cell, 1)
        max_hypervolume: Maximum hypervolume over every cell (1,)
        max_scores: Maximum values found for each score (n_scores,)
        max_sum_scores: Maximum of sum of scores (1,)
        coverage: Percentage of cells with at least one element
        number_solutions: Total number of solutions
    """

    moqd_score: MOQDScore
    max_hypervolume: MaxHypervolume
    max_scores: MaxScores
    max_sum_scores: MaxSumScores
    coverage: Coverage
    number_solutions: NumSolutions
    global_hypervolume: GlobalHypervolume


def compute_hypervolume(pareto_front: Any, reference_point: jnp.ndarray) -> jnp.ndarray:
    """
    Hypervolume computation
    TODO: implement recursive version of
    https://github.com/anyoptimization/pymoo/blob/master/pymoo/vendor/hv.py
    """

    num_objectives = pareto_front.shape[1]

    assert (
        num_objectives == 2
    ), "Hypervolume calculation for more than 2 objectives not yet supported."

    pareto_front = jnp.concatenate(
        (pareto_front, jnp.expand_dims(reference_point, axis=0)), axis=0
    )
    idx = jnp.argsort(pareto_front[:, 0])
    mask = pareto_front[idx, :] != -jnp.inf
    sorted_front = (pareto_front[idx, :] - reference_point) * mask
    sumdiff = (sorted_front[1:, 0] - sorted_front[:-1, 0]) * sorted_front[1:, 1]
    sumdiff = sumdiff * mask[:-1, 0]
    hypervolume = jnp.sum(sumdiff)

    return hypervolume


def plot_mome_pareto_fronts(
    centroids: jnp.ndarray,
    map_elites_grid: MOMERepertoire,
    maxval: float,
    minval: float,
    axes: Optional[plt.Axes] = None,
    color_style: Optional[str] = "hsv",
    with_global: Optional[bool] = False,
) -> plt.Axes:
    fitnesses = map_elites_grid.fitnesses
    grid_descriptors = map_elites_grid.descriptors

    assert fitnesses.shape[-1] == grid_descriptors.shape[-1] == 2
    assert color_style in ["hsv", "spectral"], "color_style must be hsv or spectral"

    num_centroids = len(centroids)
    grid_empty = jnp.any(fitnesses == -jnp.inf, axis=-1)

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
            cell_scores = fitnesses[i][~grid_empty[i]]
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
    axes[0].set_title("Fitness")
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
