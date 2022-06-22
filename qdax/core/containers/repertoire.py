from __future__ import annotations

import math
from functools import partial
from typing import Callable, List, Tuple, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from sklearn.cluster import KMeans

from qdax.types import Centroid, Descriptor, Fitness, Genotype, RNGKey


def compute_cvt_centroids(
    num_descriptors: int,
    num_init_cvt_samples: int,
    num_centroids: int,
    minval: Union[float, List[float]],
    maxval: Union[float, List[float]],
) -> jnp.ndarray:
    """
    Compute centroids for CVT tesselation.

    Args:
        num_descriptors: number od scalar descriptors
        num_init_cvt_samples: number of sampled point to be sued for clustering to
            determine the centroids. The larger the number of centroids and the
            number of descriptors, the higher this value must be (e.g. 100000 for
            1024 centroids and 100 descriptors).
        num_centroids: number of centroids
        minval: minimum descriptors value
        maxval: maximum descriptors value

    Returns:
        the centroids with shape (num_centroids, num_descriptors)
    """
    minval = jnp.array(minval)
    maxval = jnp.array(maxval)
    # assume here all values are in [0, 1] and rescale later
    x = np.random.rand(num_init_cvt_samples, num_descriptors)
    k_means = KMeans(
        init="k-means++",
        n_clusters=num_centroids,
        n_init=1,
    )
    k_means.fit(x)
    centroids = k_means.cluster_centers_
    # rescale now
    return jnp.asarray(centroids) * (maxval - minval) + minval


def compute_euclidean_centroids(
    num_descriptors: int,
    num_centroids: int,
    minval: Union[float, List[float]],
    maxval: Union[float, List[float]],
) -> jnp.ndarray:
    """
    Compute centroids for square Euclidean tesselation.

    Args:
        num_descriptors: number od scalar descriptors
        num_centroids: number of centroids
        minval: minimum descriptors value
        maxval: maximum descriptors value

    Returns:
        the centroids with shape (num_centroids, num_descriptors)
    """
    if num_descriptors != 2:
        raise NotImplementedError("This function supports 2 descriptors only for now.")

    sqrt_centroids = math.sqrt(num_centroids)

    if math.floor(sqrt_centroids) != sqrt_centroids:
        raise ValueError("Num centroids should be a squared number.")

    offset = 1 / (2 * int(sqrt_centroids))

    linspace = jnp.linspace(offset, 1.0 - offset, int(sqrt_centroids))
    meshes = jnp.meshgrid(linspace, linspace, sparse=False)
    centroids = jnp.stack([jnp.ravel(meshes[0]), jnp.ravel(meshes[1])], axis=-1)
    minval = jnp.array(minval)
    maxval = jnp.array(maxval)
    return jnp.asarray(centroids) * (maxval - minval) + minval


def get_cells_indices(
    batch_of_descriptors: jnp.ndarray, centroids: jnp.ndarray
) -> jnp.ndarray:
    """
    Returns the array of cells indices for a batch of descriptors
    given the centroids of the grid.

    Args:
        batch_of_descriptors: a batch of descriptors
            of shape (batch_size, num_descriptors)
        centroids: centroids array of shape (num_centroids, num_descriptors)

    Returns:
        the indices of the centroids corresponding to each vector of descriptors
            in the batch with shape (batch_size,)
    """

    def _get_cells_indices(
        descriptors: jnp.ndarray, centroids: jnp.ndarray
    ) -> jnp.ndarray:
        """
        set_of_descriptors of shape (1, num_descriptors)
        centroids of shape (num_centroids, num_descriptors)
        """
        return jnp.argmin(
            jnp.sum(jnp.square(jnp.subtract(descriptors, centroids)), axis=-1)
        )

    func = jax.vmap(lambda x: _get_cells_indices(x, centroids))
    return func(batch_of_descriptors)


class MapElitesRepertoire(flax.struct.PyTreeNode):
    """
    Class for the repertoire in Map Elites.

    Args:
        genotypes: a PyTree containing all the genotypes in the repertoire ordered
            by the centroids. Each leaf has a shape (num_centroids, num_features). The
            PyTree can be a simple Jax array or a more complex nested structure such
            as to represent parameters of neural network in Flax.
        fitnesses: an array that contains the fitness of solutions in each cell of the
            repertoire, ordered by centroids. The array shape is (num_centroids,).
        descriptors: an array that contains the descriptors of solutions in each cell
            of the repertoire, ordered by centroids. The array shape
            is (num_centroids, num_descriptors).
        centroids: an array the contains the centroids of the tesselation. The array
            shape is (num_centroids, num_descriptors).
    """

    genotypes: Genotype
    fitnesses: Fitness
    descriptors: Descriptor
    centroids: Centroid

    def save(self, path: str = "./") -> None:
        """Saves the grid on disk in the form of .npy files.

        Flattens the genotypes to store it with .npy format. Supposes that
        a user will have access to the reconstruction function when loading
        the genotypes.

        Args:
            path: Path where the data will be saved. Defaults to "./".
        """

        def flatten_genotype(genotype: Genotype) -> jnp.ndarray:
            flatten_genotype, _ = ravel_pytree(genotype)
            return flatten_genotype

        # flatten all the genotypes
        flat_genotypes = jax.vmap(flatten_genotype)(self.genotypes)

        # save data
        jnp.save(path + "genotypes.npy", flat_genotypes)
        jnp.save(path + "fitnesses.npy", self.fitnesses)
        jnp.save(path + "descriptors.npy", self.descriptors)
        jnp.save(path + "centroids.npy", self.centroids)

    @classmethod
    def load(cls, reconstruction_fn: Callable, path: str = "./") -> MapElitesRepertoire:
        """Loads a MAP Elites Grid.

        Args:
            reconstruction_fn: Function to reconstruct a PyTree
                from a flat array.
            path: Path where the data is saved. Defaults to "./".

        Returns:
            A MAP Elites Repertoire.
        """

        flat_genotypes = jnp.load(path + "genotypes.npy")
        genotypes = jax.vmap(reconstruction_fn)(flat_genotypes)

        fitnesses = jnp.load(path + "fitnesses.npy")
        descriptors = jnp.load(path + "descriptors.npy")
        centroids = jnp.load(path + "centroids.npy")

        return MapElitesRepertoire(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
        )

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:
        """
        Sample elements in the grid.

        Args:
            random_key: a jax PRNG random key
            num_samples: the number of elements to be sampled

        Returns:
            samples: a batch of genotypes sampled in the repertoire
            random_key: an updated jax PRNG random key
        """

        random_key, sub_key = jax.random.split(random_key)
        grid_empty = self.fitnesses == -jnp.inf
        p = (1.0 - grid_empty) / jnp.sum(1.0 - grid_empty)

        samples = jax.tree_map(
            lambda x: jax.random.choice(sub_key, x, shape=(num_samples,), p=p),
            self.genotypes,
        )

        return samples, random_key

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
    ) -> MapElitesRepertoire:
        """
        Add a batch of elements to the repertoire.

        Args:
            batch_of_genotypes: a batch of genotypes to be added to the repertoire.
                Similarly to the self.genotypes argument, this is a PyTree in which
                the leaves have a shape (batch_size, num_features)
            batch_of_descriptors: an array that contains the descriptors of the
                aforementioned genotypes. Its shape is (batch_size, num_descriptors)
            batch_of_fitnesses: an array that contains the fitnesses of the
                aforementioned genotypes. Its shape is (batch_size,)

        Returns:
            The updated MAP-Elites repertoire.
        """

        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)
        batch_of_fitnesses = jnp.expand_dims(batch_of_fitnesses, axis=-1)

        num_centroids = self.centroids.shape[0]

        # get fitness segment max
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices.astype(jnp.int32).squeeze(),
            num_segments=num_centroids,
        )

        cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0)

        # put dominated fitness to -jnp.inf
        batch_of_fitnesses = jnp.where(
            batch_of_fitnesses == cond_values, x=batch_of_fitnesses, y=-jnp.inf
        )

        # get addition condition
        grid_fitnesses = jnp.expand_dims(self.fitnesses, axis=-1)
        current_fitnesses = jnp.take_along_axis(grid_fitnesses, batch_of_indices, 0)
        addition_condition = batch_of_fitnesses > current_fitnesses

        # assign fake position when relevant : num_centroids is out of bound
        batch_of_indices = jnp.where(
            addition_condition, x=batch_of_indices, y=num_centroids
        )

        # create new grid
        new_grid_genotypes = jax.tree_map(
            lambda grid_genotypes, new_genotypes: grid_genotypes.at[
                batch_of_indices.squeeze()
            ].set(new_genotypes),
            self.genotypes,
            batch_of_genotypes,
        )

        # compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[batch_of_indices.squeeze()].set(
            batch_of_fitnesses.squeeze()
        )
        new_descriptors = self.descriptors.at[batch_of_indices.squeeze()].set(
            batch_of_descriptors.squeeze()
        )

        return MapElitesRepertoire(
            genotypes=new_grid_genotypes,
            fitnesses=new_fitnesses.squeeze(),
            descriptors=new_descriptors.squeeze(),
            centroids=self.centroids,
        )

    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
    ) -> MapElitesRepertoire:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Note: this function has been kept outside of the object MapElites, so it can
        be called easily called from other modules.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            fitnesses: fitness of the initial genotypes of shape (batch_size,)
            descriptors: descriptors of the initial genotypes
                of shape (batch_size, num_descriptors)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)

        Returns:
            an initialized MAP-Elite repertoire
        """

        # Initialize grid with default values
        num_centroids = centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)
        default_genotypes = jax.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape[1:]),
            genotypes,
        )
        default_descriptors = jnp.zeros(shape=(num_centroids, centroids.shape[-1]))

        repertoire = MapElitesRepertoire(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
        )

        # Add initial values to the grid
        new_repertoire = repertoire.add(genotypes, descriptors, fitnesses)

        return new_repertoire  # type: ignore
