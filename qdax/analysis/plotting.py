from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Voronoi


def get_voronoi_finite_polygons_2d(
    centroids: np.ndarray, radius: Optional[float] = None
) -> Tuple[List, np.ndarray]:
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    """
    voronoi_diagram = Voronoi(centroids)
    if voronoi_diagram.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = voronoi_diagram.vertices.tolist()

    center = voronoi_diagram.points.mean(axis=0)
    if radius is None:
        radius = voronoi_diagram.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges: Dict[jnp.ndarray, jnp.ndarray] = {}
    for (p1, p2), (v1, v2) in zip(
        voronoi_diagram.ridge_points, voronoi_diagram.ridge_vertices
    ):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(voronoi_diagram.point_region):
        vertices = voronoi_diagram.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = voronoi_diagram.points[p2] - voronoi_diagram.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = voronoi_diagram.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = voronoi_diagram.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def plot_2d_map_elites_grid(
    centroids: jnp.ndarray,
    grid_fitness: jnp.ndarray,
    minval: jnp.ndarray,
    maxval: jnp.ndarray,
    grid_descriptors: Optional[jnp.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[Figure, Axes]:
    grid_empty = grid_fitness == -jnp.inf
    num_descriptors = centroids.shape[1]
    if num_descriptors != 2:
        raise NotImplementedError("Grid plot supports 2 descriptors only for now.")

    my_cmap = cm.viridis

    fitnesses = grid_fitness
    if vmin is None:
        vmin = float(jnp.min(fitnesses[~grid_empty]))
    if vmax is None:
        vmax = float(jnp.max(fitnesses[~grid_empty]))

    # set the parameters
    font_size = 12
    params = {
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": False,
        "figure.figsize": [10, 10],
    }

    mpl.rcParams.update(params)

    # create the plot object
    if ax is None:
        fig, ax = plt.subplots(facecolor="white", edgecolor="white")

    assert (
        len(np.array(minval).shape) < 2
    ), f"minval : {minval} should be float or couple of floats"
    assert (
        len(np.array(maxval).shape) < 2
    ), f"maxval : {maxval} should be float or couple of floats"

    if len(np.array(minval).shape) == 0 and len(np.array(maxval).shape) == 0:
        ax.set_xlim(minval, maxval)
        ax.set_ylim(minval, maxval)
    else:
        ax.set_xlim(minval[0], maxval[0])
        ax.set_ylim(minval[1], maxval[1])

    ax.set(adjustable="box", aspect="equal")

    # create the regions and vertices from centroids
    regions, vertices = get_voronoi_finite_polygons_2d(centroids)

    norm = Normalize(vmin=vmin, vmax=vmax)

    # fill the plot with contours
    for region in regions:
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1)

    # fill the plot with the colors
    for idx, fitness in enumerate(fitnesses):
        if fitness > -jnp.inf:
            region = regions[idx]
            polygon = vertices[region]

            ax.fill(*zip(*polygon), alpha=0.8, color=my_cmap(norm(fitness)))

    # if descriptors are specified, add points location
    if grid_descriptors is not None:
        descriptors = grid_descriptors[~grid_empty]
        ax.scatter(
            descriptors[:, 0],
            descriptors[:, 1],
            c=fitnesses[~grid_empty],
            cmap=my_cmap,
            s=10,
            zorder=0,
        )

    # aesthetic
    ax.set_xlabel("Behavior Dimension 1")
    ax.set_ylabel("Behavior Dimension 2")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), cax=cax)
    cbar.ax.tick_params(labelsize=font_size)

    ax.set_title("MAP-Elites Grid")
    ax.set_aspect("equal")

    return fig, ax


def plot_2d_state_descriptor_archive(
    archive_data: jnp.ndarray,
    current_position: int,
    last_position: int,
    centroids: jnp.ndarray,
    minval: jnp.ndarray,
    maxval: jnp.ndarray,
    ax: Optional[plt.Axes] = None,
    save_to_path: Optional[str] = None,
) -> plt.Axes:
    num_descriptors = centroids.shape[1]
    num_state_descriptors = archive_data.shape[1]
    if num_descriptors != num_state_descriptors:
        raise NotImplementedError(
            "Need same dimension for state and behavior descriptor."
        )

    if num_state_descriptors != 2:
        raise NotImplementedError("Grid plot supports 2 descriptors only for now.")

    # set the parameters
    font_size = 12
    params = {
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": False,
        "figure.figsize": [10, 10],
    }

    mpl.rcParams.update(params)

    # create the plot object
    if ax is None:
        fig, ax = plt.subplots(facecolor="white", edgecolor="white")

    assert (
        len(np.array(minval).shape) < 2
    ), f"minval : {minval} should be float or couple of floats"
    assert (
        len(np.array(maxval).shape) < 2
    ), f"maxval : {maxval} should be float or couple of floats"

    if len(np.array(minval).shape) == 0 and len(np.array(maxval).shape) == 0:
        ax.set_xlim(minval, maxval)
        ax.set_ylim(minval, maxval)
    else:
        ax.set_xlim(minval[0], maxval[0])
        ax.set_ylim(minval[1], maxval[1])

    ax.set(adjustable="box", aspect="equal")

    # create the regions and vertices from centroids
    regions, vertices = get_voronoi_finite_polygons_2d(centroids)

    # fill the plot with contours
    for region in regions:
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1)

    ax.scatter(
        archive_data[:last_position, 0], archive_data[:last_position, 1], c="blue"
    )
    ax.scatter(
        archive_data[last_position:current_position, 0],
        archive_data[last_position:current_position, 1],
        c="red",
    )

    # aesthetic
    ax.set_xlabel("State descriptor dimension 1")
    ax.set_ylabel("State descriptor dimension 2")

    ax.set_title("State descriptor archive")
    ax.set_aspect("equal")
    if save_to_path is not None:
        fig.savefig(save_to_path)
    plt.close(fig)

    return ax


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


def plot_metrics_with_stds(
    metric: jnp.ndarray,
    steps: jnp.ndarray,
    ax: plt.Axes,
    color: str,
    label: Optional[str] = None,
) -> plt.Axes:
    """
    Plots the given metric with min, q25, mean, q75 and max
    """
    # Compute what to plot
    means = np.mean(metric, axis=0)
    q_25 = np.percentile(metric, 25, axis=0)
    q_75 = np.percentile(metric, 75, axis=0)
    mins = np.min(metric, axis=0)
    maxs = np.max(metric, axis=0)

    # Fill between percentiles
    ax.fill_between(steps, q_25, q_75, alpha=0.5, facecolor=color)
    ax.fill_between(steps, mins, maxs, alpha=0.1, facecolor=color)

    # Plot the mean
    ax.plot(steps, means, color=color, linewidth=2, label=label)

    return ax


def plot_global_pareto_front(
    pareto_front: jnp.ndarray,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    colour: Optional[str] = None,
) -> Union[plt.Axes, Tuple[Any, plt.Axes]]:
    """
    Plots the global Pareto Front
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], color=colour, label=label)
        return fig, ax
    else:
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], color=colour, label=label)

    return ax


def plot_global_pareto_descriptors(
    pareto_descriptors: jnp.ndarray,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    colour: Optional[str] = None,
) -> Union[plt.Axes, Tuple[Any, plt.Axes]]:
    """
    Plots the global Pareto descriptors
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            pareto_descriptors[:, 0],
            pareto_descriptors[:, 1],
            color=colour,
            label=label,
        )
        return fig, ax
    else:

        ax.scatter(
            pareto_descriptors[:, 0],
            pareto_descriptors[:, 1],
            color=colour,
            label=label,
        )

    return ax


def multiline(
    xs: Iterable, ys: Iterable, c: Iterable, ax: Axes = None, **kwargs: Any
) -> LineCollection:
    """Plot lines with different colorings (with c a container of numbers mapped to
    colormap)

    Note:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def save_skill_trajectory(
    trajectories: jnp.ndarray,
    skills: jnp.ndarray,
    min_values: jnp.ndarray,
    max_values: jnp.ndarray,
) -> Tuple[Figure, Axes]:
    num_skills = skills.shape[1]
    c = skills.argmax(axis=1)
    fig, ax = plt.subplots()
    # rot = jnp.array([[0, 1], [-1, 0]])
    xs, ys = trajectories
    lc = multiline(xs=xs, ys=ys, c=c, ax=ax, cmap="gist_rainbow")
    ax.set_ylim(min_values[1], max_values[1])
    ax.set_xlim(min_values[0], max_values[0])
    ax.set_xlabel("Behavior Dimension 1")
    ax.set_ylabel("Behavior Dimension 2")
    ax.set_aspect("equal")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    axcb = fig.colorbar(lc, cax=cax)
    axcb.set_ticks(np.arange(num_skills, dtype=int))
    ax.set_title("Skill trajectories")
    return fig, ax
