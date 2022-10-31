from typing import Any, Dict, Iterable, List, Optional, Tuple

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

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.containers.mome_repertoire import MOMERepertoire


def get_voronoi_finite_polygons_2d(
    centroids: np.ndarray, radius: Optional[float] = None
) -> Tuple[List, np.ndarray]:
    """Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions."""
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


def plot_2d_map_elites_repertoire(
    centroids: jnp.ndarray,
    repertoire_fitnesses: jnp.ndarray,
    minval: jnp.ndarray,
    maxval: jnp.ndarray,
    repertoire_descriptors: Optional[jnp.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[Optional[Figure], Axes]:
    """Plot a visual representation of a 2d map elites repertoire.

    TODO: Use repertoire as input directly. Because this
    function is very specific to repertoires.

    Args:
        centroids: the centroids of the repertoire
        repertoire_fitnesses: the fitness of the repertoire
        minval: minimum values for the descritors
        maxval: maximum values for the descriptors
        repertoire_descriptors: the descriptors. Defaults to None.
        ax: a matplotlib axe for the figure to plot. Defaults to None.
        vmin: minimum value for the fitness. Defaults to None. If not given,
            the value will be set to the minimum fitness in the repertoire.
        vmax: maximum value for the fitness. Defaults to None. If not given,
            the value will be set to the maximum fitness in the repertoire.

    Raises:
        NotImplementedError: does not work for descriptors dimension different
        from 2.

    Returns:
        A figure and axes object, corresponding to the visualisation of the
        repertoire.
    """

    # TODO: check it and fix it if needed
    grid_empty = repertoire_fitnesses == -jnp.inf
    num_descriptors = centroids.shape[1]
    if num_descriptors != 2:
        raise NotImplementedError("Grid plot supports 2 descriptors only for now.")

    my_cmap = cm.viridis

    fitnesses = repertoire_fitnesses
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
    fig = None
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
    if repertoire_descriptors is not None:
        descriptors = repertoire_descriptors[~grid_empty]
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


def plot_map_elites_results(
    env_steps: jnp.ndarray,
    metrics: Dict,
    repertoire: MapElitesRepertoire,
    min_bd: jnp.ndarray,
    max_bd: jnp.ndarray,
) -> Tuple[Optional[Figure], Axes]:
    """Plots three usual QD metrics, namely the coverage, the maximum fitness
    and the QD-score, along the number of environment steps. This function also
    plots a visualisation of the final map elites grid obtained. It ensures that
    those plots are aligned together to give a simple and efficient visualisation
    of an optimization process.

    Args:
        env_steps: the array containing the number of steps done in the environment.
        metrics: a dictionary containing metrics from the optimizatoin process.
        repertoire: the final repertoire obtained.
        min_bd: the mimimal possible values for the bd.
        max_bd: the maximal possible values for the bd.

    Returns:
        A figure and axes with the plots of the metrics and visualisation of the grid.
    """
    # Customize matplotlib params
    font_size = 16
    params = {
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": False,
        "axes.titlepad": 10,
    }

    mpl.rcParams.update(params)

    # Visualize the training evolution and final repertoire
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(40, 10))

    # env_steps = jnp.arange(num_iterations) * episode_length * batch_size

    axes[0].plot(env_steps, metrics["coverage"])
    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Coverage in %")
    axes[0].set_title("Coverage evolution during training")
    axes[0].set_aspect(0.95 / axes[0].get_data_ratio(), adjustable="box")

    axes[1].plot(env_steps, metrics["max_fitness"])
    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Maximum fitness")
    axes[1].set_title("Maximum fitness evolution during training")
    axes[1].set_aspect(0.95 / axes[1].get_data_ratio(), adjustable="box")

    axes[2].plot(env_steps, metrics["qd_score"])
    axes[2].set_xlabel("Environment steps")
    axes[2].set_ylabel("QD Score")
    axes[2].set_title("QD Score evolution during training")
    axes[2].set_aspect(0.95 / axes[2].get_data_ratio(), adjustable="box")

    _, axes = plot_2d_map_elites_repertoire(
        centroids=repertoire.centroids,
        repertoire_fitnesses=repertoire.fitnesses,
        minval=min_bd,
        maxval=max_bd,
        repertoire_descriptors=repertoire.descriptors,
        ax=axes[3],
    )

    return fig, axes


def multiline(
    xs: Iterable, ys: Iterable, c: Iterable, ax: Optional[Axes] = None, **kwargs: Any
) -> LineCollection:
    """Plot lines with different colorings (with c a container of numbers mapped to
        colormap)

    Note:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Args:
        xs: First dimension of the trajectory.
        ys: Second dimension of the trajectory.
        c: Colors - one for each trajectory.
        ax: A matplotlib axe. Defaults to None.

    Returns:
        Return a collection of lines corresponding to the trajectories.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    # Note: error if c is given as a list here.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    # Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def plot_skills_trajectory(
    trajectories: jnp.ndarray,
    skills: jnp.ndarray,
    min_values: jnp.ndarray,
    max_values: jnp.ndarray,
) -> Tuple[Figure, Axes]:
    """Plots skills trajectories on a single plot with
    different colors to recognize the skills.

    The plot can contain several trajectories of the same
    skill.

    Args:
        trajectories: skills trajectories
        skills: skills corresponding to the given trajectories
        min_values: minimum values that can be taken by the steps
            of the trajectory
        max_values: maximum values that can be taken by the steps
            of the trajectory

    Returns:
        A figure and axes.
    """
    # get number of skills
    num_skills = skills.shape[1]

    # create color from possible skills (indexed from 0 to num skills - 1)
    c = skills.argmax(axis=1)

    # create the figure
    fig, ax = plt.subplots()

    # get lines from the trajectories
    xs, ys = trajectories
    lc = multiline(xs=xs, ys=ys, c=c, ax=ax, cmap="gist_rainbow")

    # set aesthetics
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


def plot_mome_pareto_fronts(
    centroids: jnp.ndarray,
    repertoire: MOMERepertoire,
    maxval: float,
    minval: float,
    axes: Optional[plt.Axes] = None,
    color_style: Optional[str] = "hsv",
    with_global: Optional[bool] = False,
) -> plt.Axes:
    """Plot the pareto fronts from all cells of the mome repertoire.

    Args:
        centroids: centroids of the repertoire
        repertoire: mome repertoire
        maxval: maximum values for the descriptors
        minval: minimum values for the descriptors
        axes: matplotlib axes. Defaults to None.
        color_style: style of the colors used. Defaults to "hsv".
        with_global: plot the global pareto front in addition.
            Defaults to False.

    Returns:
        Returns the axes object with the plot.
    """
    fitnesses = repertoire.fitnesses
    repertoire_descriptors = repertoire.descriptors

    assert fitnesses.shape[-1] == repertoire_descriptors.shape[-1] == 2
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
            cell = repertoire_descriptors[i][~grid_empty[i]]
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
        global_pareto_front, pareto_bool = repertoire.compute_global_pareto_front()
        global_pareto_descriptors = jnp.concatenate(repertoire_descriptors)[pareto_bool]
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


def vector_to_rgb(angle: float, absolute: float) -> Any:
    """Returns a color based on polar coordinates.

    Args:
        angle: a given angle
        absolute: a ref

    Returns:
        An appropriate color.
    """

    # normalize angle
    angle = angle % (2 * np.pi)
    if angle < 0:
        angle += 2 * np.pi

    # rise absolute to avoid black colours
    absolute = (absolute + 0.5) / 1.5

    return mpl.colors.hsv_to_rgb((angle / 2 / np.pi, 1, absolute))


def plot_global_pareto_front(
    pareto_front: jnp.ndarray,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> Tuple[Optional[Figure], plt.Axes]:
    """Plots the global Pareto Front.

    Args:
        pareto_front: a pareto front
        ax: a matplotlib ax. Defaults to None.
        label: a given label. Defaults to None.
        color: a color for the plotted points. Defaults to None.

    Returns:
        A figure and an axe.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], color=color, label=label)
        return fig, ax
    else:
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], color=color, label=label)

    return fig, ax


def _get_projection_in_1d(
    integer_coordinates: jnp.ndarray, bases_tuple: Tuple[int, ...]
) -> int:
    """Converts an integer vector into a single integer,
    given tuple of bases to consider for conversion.

    This conversion is ensured to be unique, provided that
    for all index i: x[i] < bases_tuple[i].
    The vector and tuple of bases must have the same length.

    For example if x=jnp.array([3, 1, 2]) and the bases are (5, 7, 3).
    then the projection is 3*(7*3) + 1*(3) + 2 = 47.

    Args:
        integer_coordinates: the coordinates of the points (should be integers).
        bases_tuple: the bases to use.

    Returns:
        The projection of the point in 1D (int).
    """
    assert jnp.size(integer_coordinates) == len(
        bases_tuple
    ), "x should have the same size as bases"

    integer_coordinates = integer_coordinates.ravel().tolist()

    # build the conversion
    coordinate = 0
    for x_coord, base in zip(integer_coordinates, bases_tuple):
        coordinate = coordinate * base + x_coord

    return coordinate


def _get_projection_in_2d(
    integer_coordinates: jnp.ndarray, bases: Tuple[int, ...]
) -> Tuple[int, int]:
    """Projects an integer vector into a pair of integers,
    (given tuple of bases to consider for conversion).

    For example if x=jnp.array([3, 1, 2, 5]) and the bases are (5, 2, 3, 7).
    then the projection is obtained by:
    - projecting in 1D the point jnp.array([3, 2]) with the bases (5, 3)
    - projecting in 1D the point jnp.array([1, 5]) with the bases (2, 7)

    Args:
        integer_coordinates: the coordinates of the points (should be integers).
        bases_tuple: the bases to use.

    Returns:
        The projection of the point in 2D (pair of integers).
    """
    integer_coordinates = integer_coordinates.ravel()
    x0 = _get_projection_in_1d(integer_coordinates[::2], bases[::2])
    x1 = _get_projection_in_1d(integer_coordinates[1::2], bases[1::2])
    return x0, x1


def plot_multidimensional_map_elites_grid(
    repertoire: MapElitesRepertoire,
    minval: jnp.ndarray,
    maxval: jnp.ndarray,
    grid_shape: Tuple[int, ...],
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[Optional[Figure], Axes]:
    """Plot a visual 2D representation of a multidimensional MAP-Elites repertoire
    (where the dimensionality of descriptors can be greater than 2).
    Args:
        repertoire: the MAP-Elites repertoire to plot.
        minval: minimum values for the descriptors
        maxval: maximum values for the descriptors
        grid_shape: the resolution of the grid.
        ax: a matplotlib axe for the figure to plot. Defaults to None.
        vmin: minimum value for the fitness. Defaults to None.
        vmax: maximum value for the fitness. Defaults to None.
    Raises:
        ValueError: the resolution should be an int or a tuple
    Returns:
        A figure and axes object, corresponding to the visualisation of the
        repertoire.
    """

    descriptors = repertoire.descriptors
    fitnesses = repertoire.fitnesses

    is_grid_empty = fitnesses.ravel() == -jnp.inf
    num_descriptors = descriptors.shape[1]

    if isinstance(grid_shape, tuple):
        assert (
            len(grid_shape) == num_descriptors
        ), "grid_shape should have the same length as num_descriptors"
    else:
        raise ValueError("resolution should be a tuple")

    assert np.size(minval) == num_descriptors or np.size(minval) == 1, (
        f"minval : {minval} should either be of size 1 "
        f"or have the same size as the number of descriptors: {num_descriptors}"
    )
    assert np.size(maxval) == num_descriptors or np.size(maxval) == 1, (
        f"maxval : {maxval} should either be of size 1 "
        f"or have the same size as the number of descriptors: {num_descriptors}"
    )

    non_empty_descriptors = descriptors[~is_grid_empty]
    non_empty_fitnesses = fitnesses[~is_grid_empty]

    # convert the descriptors to integer coordinates, depending on the resolution.
    resolutions_array = jnp.array(grid_shape)
    descriptors_integers = jnp.asarray(
        jnp.floor(
            resolutions_array * (non_empty_descriptors - minval) / (maxval - minval)
        ),
        dtype=jnp.int32,
    )

    # total number of grid cells along each dimension of the grid
    size_grid_x = np.prod(np.array(grid_shape[0::2]))
    # convert to int for the 1d case - if not, value 1.0 is given
    size_grid_y = np.prod(np.array(grid_shape[1::2]), dtype=int)

    # initialise the grid
    grid_2d = np.full(
        (size_grid_x.item(), size_grid_y.item()),
        fill_value=jnp.nan,
    )

    # put solutions in the grid according to their projected 2-dimensional coordinates
    for desc, fit in zip(descriptors_integers, non_empty_fitnesses):
        projection_2d = _get_projection_in_2d(desc, grid_shape)
        if jnp.isnan(grid_2d[projection_2d]) or fit.item() > grid_2d[projection_2d]:
            grid_2d[projection_2d] = fit.item()

    # set plot parameters
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
    fig = None
    if ax is None:
        fig, ax = plt.subplots(facecolor="white", edgecolor="white")
    ax.set(adjustable="box", aspect="equal")

    my_cmap = cm.viridis

    if vmin is None:
        vmin = float(jnp.min(non_empty_fitnesses))
    if vmax is None:
        vmax = float(jnp.max(non_empty_fitnesses))

    ax.imshow(
        grid_2d.T,
        origin="lower",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        cmap=my_cmap,
    )

    # aesthetic
    ax.set_xlabel("Behavior Dimension 1")
    ax.set_ylabel("Behavior Dimension 2")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), cax=cax)
    cbar.ax.tick_params(labelsize=font_size)

    ax.set_title("MAP-Elites Grid")
    ax.set_aspect("equal")

    def _get_ticks_positions(
        total_size_grid_axis: int, step_ticks_on_axis: int
    ) -> jnp.ndarray:
        """
        Get the positions of the ticks on the grid axis.
        Args:
            total_size_grid_axis: total size of the grid axis
            step_ticks_on_axis: step of the ticks
        Returns:
            The positions of the ticks on the plot.
        """
        return np.arange(0, total_size_grid_axis + 1, step_ticks_on_axis) - 0.5

    # Ticks position
    major_ticks_x = _get_ticks_positions(
        size_grid_x.item(), step_ticks_on_axis=np.prod(grid_shape[2::2]).item()
    )
    minor_ticks_x = _get_ticks_positions(
        size_grid_x.item(), step_ticks_on_axis=np.prod(grid_shape[4::2]).item()
    )
    major_ticks_y = _get_ticks_positions(
        size_grid_y.item(), step_ticks_on_axis=np.prod(grid_shape[3::2]).item()
    )
    minor_ticks_y = _get_ticks_positions(
        size_grid_y.item(), step_ticks_on_axis=np.prod(grid_shape[5::2]).item()
    )

    ax.set_xticks(
        major_ticks_x,
    )
    ax.set_xticks(
        minor_ticks_x,
        minor=True,
    )
    ax.set_yticks(
        major_ticks_y,
    )
    ax.set_yticks(
        minor_ticks_y,
        minor=True,
    )

    # Ticks aesthetics
    ax.tick_params(
        which="minor",
        color="gray",
        labelcolor="gray",
        size=5,
    )
    ax.tick_params(
        which="major",
        labelsize=font_size,
        size=7,
    )

    ax.grid(which="minor", alpha=1.0, color="#000000", linewidth=0.5)
    if len(grid_shape) > 2:
        ax.grid(which="major", alpha=1.0, color="#000000", linewidth=2.5)

    def _get_positions_labels(
        _minval: float, _maxval: float, _number_ticks: int, _step_labels_ticks: int
    ) -> List[str]:
        positions = jnp.linspace(_minval, _maxval, num=_number_ticks)

        list_str_positions = []
        for index_tick, position in enumerate(positions):
            if index_tick % _step_labels_ticks != 0:
                character = ""
            else:
                character = f"{position:.2E}"
            list_str_positions.append(character)
        # forcing the last tick label
        list_str_positions[-1] = f"{positions[-1]:.2E}"
        return list_str_positions

    number_label_ticks = 4

    if len(major_ticks_x) // number_label_ticks > 0:
        ax.set_xticklabels(
            _get_positions_labels(
                _minval=minval[0],
                _maxval=maxval[0],
                _number_ticks=len(major_ticks_x),
                _step_labels_ticks=len(major_ticks_x) // number_label_ticks,
            )
        )
    if len(major_ticks_y) // number_label_ticks > 0:
        ax.set_yticklabels(
            _get_positions_labels(
                _minval=minval[1],
                _maxval=maxval[1],
                _number_ticks=len(major_ticks_y),
                _step_labels_ticks=len(major_ticks_y) // number_label_ticks,
            )
        )

    return fig, ax
