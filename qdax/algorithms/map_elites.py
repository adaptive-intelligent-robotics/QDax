"""Core components of the MAP-Elites algorithm."""
import math
from functools import partial
from typing import Callable, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.cluster import KMeans

from qdax.types import (
    Centroid,
    Descriptor,
    EmitterState,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)


def compute_cvt_centroids(
    num_descriptors: int,
    num_init_cvt_samples: int,
    num_centroids: int,
    minval: float,
    maxval: float,
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
    minval: float,
    maxval: float,
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

    linspace = jnp.linspace(minval, maxval, int(sqrt_centroids))
    meshes = jnp.meshgrid(linspace, linspace, sparse=False)
    centroids = jnp.stack([jnp.ravel(meshes[0]), jnp.ravel(meshes[1])], axis=-1)
    return centroids


def get_cells_indices(batch_of_descriptors: jnp.ndarray, centroids: jnp.ndarray):
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


@flax.struct.dataclass
class MapElitesRepertoire:
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

    def save(self, path="./"):
        """
        Save the grid on disk in the form of .npy files.

        TODO: update genotypes saving
        """
        jnp.save(path + "genotypes.npy", self.genotypes)
        jnp.save(path + "fitnesses.npy", self.fitnesses)
        jnp.save(path + "descriptors.npy", self.descriptors)
        jnp.save(path + "centroids.npy", self.centroids)

    # TODO: might need to be modified
    @classmethod
    def load(cls, path="./"):
        """Load a MAP Elites Grid
        TODO: update genotypes loading

        """

        genotypes = jnp.load(path + "genotypes.npy")
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
        p = (1.0 - grid_empty) / jnp.sum(grid_empty)

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
    ) -> "MapElitesRepertoire":
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
        new_grid_genotypes = jax.tree_multimap(
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

        return self.replace(
            genotypes=new_grid_genotypes,
            fitnesses=new_fitnesses.squeeze(),
            descriptors=new_descriptors.squeeze(),
        )


@jax.jit
def init_map_elites_repertoire(
    genotypes: Genotype,
    fitnesses: Fitness,
    descriptors: Descriptor,
    centroids: Centroid,
) -> MapElitesRepertoire:
    """
    Initialize a Map-Elites repertoire with an initial population of genotypes. Requires
    the definition of centroids that can be computed with any method such as
    CVT or Euclidean mapping.

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
    repertoire = repertoire.add(genotypes, descriptors, fitnesses)

    return repertoire


class MAPElites:
    """
    Core elements of the MAP-Elites algorithm.

    Args:
        scoring_function: a function that takes a batch of genotypes and compute
            their fitnesses and descriptors
        emitter_function: an emitter function that takes the MAP-Elites repertoire,
            sample elements in it, clone them and apply transformations to get a
            new population ready to be evaluated. This function may also update an
            internal state called emitter state.
        metrics_function: a function that takes a MAP-Elites repertoire and compute
            any useful metric to track its evolution
    """

    def __init__(
        self,
        scoring_function: Callable[[Genotype], Tuple[Fitness, Descriptor]],
        emitter_function: Callable[
            [MapElitesRepertoire, EmitterState, RNGKey],
            Tuple[Genotype, EmitterState, RNGKey],
        ],
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
    ):
        self._scoring_function = scoring_function
        self._emitter_function = emitter_function
        self._metrics_function = metrics_function

    @partial(jax.jit, static_argnames=("self",))
    def init_fn(
        self,
        init_genotypes: Genotype,
        centroids: Centroid,
    ) -> MapElitesRepertoire:
        """
        Initialize a Map-Elites grid with an initial population of genotypes. Requires
        the definition of centroids that can be computed with any method such as
        CVT or Euclidean mapping.

        Args:
            init_genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)

        Returns:
            an initialized MAP-Elite repertoire
        """
        init_fitnesses, init_descriptors = self._scoring_function(init_genotypes)

        grid = init_map_elites_repertoire(
            genotypes=init_genotypes,
            fitnesses=init_fitnesses,
            descriptors=init_descriptors,
            centroids=centroids,
        )
        return grid

    @partial(jax.jit, static_argnames=("self",))
    def update_fn(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: EmitterState,
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, EmitterState, Metrics, RNGKey]:
        """
        Performs one iteration of the MAP-Elites algorithm.
        1. A batch of genotypes is sampled in the archive and the genotypes are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the archive.

        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Results:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """

        genotypes, emitter_state, random_key = self._emitter_function(
            repertoire, emitter_state, random_key
        )
        fitnesses, descriptors = self._scoring_function(genotypes)
        repertoire = repertoire.add(genotypes, descriptors, fitnesses)
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics, random_key
