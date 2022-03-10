"""Core components of the MAP-Elites algorithm."""
import dataclasses
import math
from functools import partial
from typing import Callable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import scan
from qdax.algorithms.types import (
    Centroids,
    Descriptors,
    Fitness,
    Genotypes,
    Grid,
    GridDescriptors,
    GridScores,
    RNGKey,
    Scores,
)
from sklearn.cluster import KMeans


def compute_cvt_centroids(
    num_descriptors: int,
    num_init_cvt_samples: int,
    num_centroids: int,
    minval: float,
    maxval: float,
) -> jnp.ndarray:
    """
    Compute centroids for CVT grid.
    """
    minval = jnp.array(minval)
    maxval = jnp.array(maxval)
    # all values are in [0, 1]
    x = np.random.rand(num_init_cvt_samples, num_descriptors)
    k_means = KMeans(
        init="k-means++",
        n_clusters=num_centroids,
        n_init=1,
    )
    k_means.fit(x)
    centroids = k_means.cluster_centers_
    return jnp.asarray(centroids) * (maxval - minval) + minval


def compute_euclidean_centroids(
    num_descriptors: int,
    num_centroids: int,
    minval: float,
    maxval: float,
) -> jnp.ndarray:
    """
    Compute centroids for square Euclidean grid.
    """
    if num_descriptors != 2:
        raise NotImplementedError("This function supports 2 descriptors only for now.")

    sqrt_centroids = math.sqrt(num_centroids)

    if math.floor(sqrt_centroids) != sqrt_centroids:
        raise ValueError("Num centroids should be a squared number.")

    linspace = jnp.linspace(minval, maxval, int(sqrt_centroids))
    meshes = jnp.meshgrid(linspace, linspace, sparse=False)
    centroids = jnp.stack([jnp.ravel(meshes[0]), jnp.ravel(meshes[1])], axis=-1)
    return centroids


def get_cells_indices(batch_of_descriptors: jnp.ndarray, centroids: jnp.ndarray):
    """
    Returns the array of cells indices for a batch of descriptors
    given the centroids of the grid.

    batch_of_descriptors of shape (batch_size, num_descriptors)
    centroids of shape (num_centroids, num_descriptors)
    """

    def _get_cells_indices(descriptors: jnp.ndarray, centroids: jnp.ndarray):
        """
        set_of_descriptors of shape (1, num_descriptors)
        centroids of shape (num_centroids, num_descriptors)
        """
        return jnp.argmin(
            jnp.sum(jnp.square(jnp.subtract(descriptors, centroids)), axis=-1)
        )

    func = jax.vmap(lambda x: _get_cells_indices(x, centroids))
    return func(batch_of_descriptors)


@dataclasses.dataclass
class MapElitesGrid:
    """Class for the grid in Map Elites"""

    grid: Grid
    grid_scores: GridScores
    grid_descriptors: GridDescriptors
    centroids: Centroids

    def save(self, path="./"):
        """
        Save the grid on disk in the form of .npy files.
        """
        jnp.save(path + "grid.npy", self.grid)
        jnp.save(path + "grid_scores.npy", self.grid_scores)
        jnp.save(path + "grid_descriptors.npy", self.grid_descriptors)
        jnp.save(path + "centroids.npy", self.centroids)

    @classmethod
    def load(cls, path="./"):
        """Load a MAP Elites Grid"""

        grid_genotypes = jnp.load(path + "grid.npy")
        grid_scores = jnp.load(path + "grid_scores.npy")
        grid_descriptors = jnp.load(path + "grid_descriptors.npy")
        centroids = jnp.load(path + "centroids.npy")

        return MapElitesGrid(
            grid=grid_genotypes,
            grid_scores=grid_scores,
            grid_descriptors=grid_descriptors,
            centroids=centroids,
        )

    def sample_in_grid(
        self, random_key: RNGKey, num_samples: int
    ) -> Tuple[Genotypes, RNGKey]:
        """
        Sample elements in the grid.
        """
        # grid_x of shape (num_centroids, genotype_dim)
        # grid_empty of shape (num_centroids,),
        # vector of boolean True if cell empty False otherwise

        random_key, sub_key = jax.random.split(random_key)
        grid_empty = self.grid_scores == -jnp.inf
        p = (1.0 - grid_empty) / jnp.sum(grid_empty)
        return (
            jax.random.choice(sub_key, self.grid, shape=(num_samples,), p=p),
            random_key,
        )

    @jax.jit
    def add_in_grid(
        self,
        batch_of_genotypes: Genotypes,
        batch_of_descriptors: Descriptors,
        batch_of_fitnesses: Scores,
    ) -> "MapElitesGrid":
        """
        Insert a batch of elements in the grid.

        grid of shape (num_centroids,)
        vector of booleans True if cell empty False otherwise
        batch_of_descriptors of shape (batch_size, num_descriptors)
        batch_of_fitnesses of shape (batch_size,)
        """

        desc_dim = batch_of_descriptors.shape[1]
        genotype_dim = batch_of_genotypes.shape[1]

        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)
        batch_of_fitnesses = jnp.expand_dims(batch_of_fitnesses, axis=-1)

        concat_array = jnp.concatenate(
            [
                batch_of_genotypes,
                batch_of_descriptors,
                batch_of_fitnesses,
                batch_of_indices,
            ],
            axis=-1,
        )

        def _add_one(carry: "MapElitesGrid", x: jnp.ndarray):
            """Single insertion"""
            genotype, descriptors, fitness, index = jnp.split(
                jnp.ravel(x),
                [genotype_dim, genotype_dim + desc_dim, genotype_dim + desc_dim + 1],
            )
            index = index.astype(jnp.int32)
            # genotype, descriptors, fitness, index = x
            cond = fitness > self.grid_scores[index]

            genotype_cell = jnp.where(cond, genotype, carry.grid[index])
            fitness_cell = jnp.where(cond, fitness, carry.grid_scores[index])
            desc_cell = jnp.where(cond, descriptors, carry.grid_descriptors[index])

            carry.grid = carry.grid.at[index].set(genotype_cell)
            carry.grid_scores = carry.grid_scores.at[index].set(fitness_cell)
            carry.grid_descriptors = carry.grid_descriptors.at[index].set(desc_cell)

            return carry, ()

        self, _ = scan(_add_one, self, concat_array)

        return self


jax.tree_util.register_pytree_node(
    MapElitesGrid,
    lambda s: ((s.grid, s.grid_scores, s.grid_descriptors, s.centroids), None),
    lambda _, xs: MapElitesGrid(xs[0], xs[1], xs[2], xs[3]),
)


@dataclasses.dataclass
class GridInfo:
    """
    Minimal information to specify a tesselation
    """

    descriptors_range: List
    extreme_values_range: List
    centroids_type: str = "cvt"
    num_centroids: int = 100


@dataclasses.dataclass
class QDMetrics:
    """Class to store QD performance metrics."""

    qd_score: jnp.ndarray
    max_fitness: jnp.ndarray
    coverage: jnp.ndarray

    def concatenate(self, other):

        return QDMetrics(
            jnp.concatenate((self.qd_score, other.qd_score)),
            jnp.concatenate((self.max_fitness, other.max_fitness)),
            jnp.concatenate((self.coverage, other.coverage)),
        )


jax.tree_util.register_pytree_node(
    QDMetrics,
    lambda s: ((s.qd_score, s.max_fitness, s.coverage), None),
    lambda _, xs: QDMetrics(xs[0], xs[1], xs[2]),
)


class MAPElites:
    """Core elements of the MAP-Elites algorithm."""

    def __init__(
        self,
        scoring_function: Callable[[Genotypes], Tuple[Fitness, Descriptors]],
        crossover_function: Callable[
            [Genotypes, Genotypes, RNGKey], Tuple[Genotypes, RNGKey]
        ],
        mutation_function: Callable[[Genotypes, RNGKey], Tuple[Genotypes, RNGKey]],
        crossover_percentage: float,
    ):
        self._scoring_function = scoring_function
        self._crossover_function = crossover_function
        self._mutation_function = mutation_function
        self._crossover_percentage = crossover_percentage

    @partial(jax.jit, static_argnames=("self",))
    def init_fn(
        self,
        init_genotypes: Genotypes,
        centroids: Centroids,
    ) -> MapElitesGrid:
        """
        Initialize a Map-Elites grid with an initial population of genotypes. Requires
        the definition of centroids that can be computed with any method such as
        CVT or Euclidean mapping.
        """
        init_fitnesses, init_descriptors = self._scoring_function(init_genotypes)

        num_centroids = centroids.shape[0]
        grid_scores = -jnp.inf * jnp.ones(shape=num_centroids)
        grid_genotypes = jnp.zeros(shape=(num_centroids,) + init_genotypes.shape[1:])
        grid_descriptors = jnp.zeros(shape=(num_centroids, init_descriptors.shape[-1]))

        grid = MapElitesGrid(
            grid=grid_genotypes,
            grid_scores=grid_scores,
            grid_descriptors=grid_descriptors,
            centroids=centroids,
        )
        grid = grid.add_in_grid(init_genotypes, init_descriptors, init_fitnesses)

        return grid

    @partial(jax.jit, static_argnames=("self", "batch_size"))
    def update_fn(
        self, grid: MapElitesGrid, random_key: RNGKey, batch_size: int
    ) -> Tuple[MapElitesGrid, RNGKey]:
        """
        Performs one iteration of the MAP-Elites algorithm.
        1. A batch of genotypes is sampled in the archive and the genotypes are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the archive.
        """
        n_crossover = int(batch_size * self._crossover_percentage)
        n_mutation = batch_size - n_crossover

        if n_crossover > 0:
            x1, random_key = grid.sample_in_grid(random_key, n_crossover)
            x2, random_key = grid.sample_in_grid(random_key, n_crossover)

            x_crossover, random_key = self._crossover_function(x1, x2, random_key)

        if n_mutation > 0:
            x1, random_key = grid.sample_in_grid(random_key, n_mutation)
            x_mutation, random_key = self._mutation_function(x1, random_key)

        if n_crossover == 0:
            genotypes = x_mutation
        elif n_mutation == 0:
            genotypes = x_crossover
        else:
            genotypes = jnp.concatenate([x_crossover, x_mutation], axis=0)

        fitnesses, descriptors = self._scoring_function(genotypes)
        grid = grid.add_in_grid(genotypes, descriptors, fitnesses)

        return grid, random_key
