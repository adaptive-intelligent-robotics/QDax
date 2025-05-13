"""This file contains util functions and a class to define
a repertoire, used to store individuals in the MAP-Elites
algorithm as well as several variants."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from numpy.random import RandomState
from sklearn.cluster import KMeans

from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.emitters.repertoire_selectors.selector import (
    MapElitesRepertoireT,
    Selector,
)
from qdax.core.emitters.repertoire_selectors.uniform_selector import UniformSelector
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    RNGKey,
)


def compute_cvt_centroids(
    num_descriptors: int,
    num_init_cvt_samples: int,
    num_centroids: int,
    minval: Union[float, List[float]],
    maxval: Union[float, List[float]],
    key: RNGKey,
) -> Centroid:
    """Compute centroids for CVT tessellation.

    Args:
        num_descriptors: number of scalar descriptors
        num_init_cvt_samples: number of sampled point to be sued for clustering to
            determine the centroids. The larger the number of centroids and the
            number of descriptors, the higher this value must be (e.g. 100000 for
            1024 centroids and 100 descriptors).
        num_centroids: number of centroids
        minval: minimum descriptors value
        maxval: maximum descriptors value
        key: a jax PRNG random key

    Returns:
        the centroids with shape (num_centroids, num_descriptors)
        key: an updated jax PRNG random key
    """
    minval = jnp.array(minval)
    maxval = jnp.array(maxval)

    # assume here all values are in [0, 1] and rescale later
    key, subkey = jax.random.split(key)
    x = jax.random.uniform(key=subkey, shape=(num_init_cvt_samples, num_descriptors))

    # compute k means
    k_means = KMeans(
        init="k-means++",
        n_clusters=num_centroids,
        n_init=1,
        random_state=RandomState(jax.random.key_data(key)),
    )
    k_means.fit(x)
    centroids = k_means.cluster_centers_
    # rescale now
    return jnp.asarray(centroids) * (maxval - minval) + minval


def compute_euclidean_centroids(
    grid_shape: Tuple[int, ...],
    minval: Union[float, List[float]],
    maxval: Union[float, List[float]],
) -> jnp.ndarray:
    """Compute centroids for square Euclidean tessellation.

    Args:
        grid_shape: number of centroids per descriptor dimension
        minval: minimum descriptors value
        maxval: maximum descriptors value

    Returns:
        the centroids with shape (num_centroids, num_descriptors)
    """
    # get number of descriptors
    num_descriptors = len(grid_shape)

    # prepare list of linspaces
    linspace_list = []
    for num_centroids_in_dim in grid_shape:
        offset = 1 / (2 * num_centroids_in_dim)
        linspace = jnp.linspace(offset, 1.0 - offset, num_centroids_in_dim)
        linspace_list.append(linspace)

    meshes = jnp.meshgrid(*linspace_list, sparse=False)

    # create centroids
    centroids = jnp.stack(
        [jnp.ravel(meshes[i]) for i in range(num_descriptors)], axis=-1
    )
    minval = jnp.array(minval)
    maxval = jnp.array(maxval)
    return jnp.asarray(centroids) * (maxval - minval) + minval


def get_cells_indices(
    batch_of_descriptors: jnp.ndarray, centroids: jnp.ndarray
) -> jnp.ndarray:
    """
    Returns the array of cells indices for a batch of descriptors
    given the centroids of the repertoire.

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
        """Set_of_descriptors of shape (1, num_descriptors)
        centroids of shape (num_centroids, num_descriptors)
        """
        return jnp.argmin(
            jnp.sum(jnp.square(jnp.subtract(descriptors, centroids)), axis=-1)
        )

    func = jax.vmap(lambda x: _get_cells_indices(x, centroids))
    return func(batch_of_descriptors)


class MapElitesRepertoire(GARepertoire):
    """Class for the repertoire in Map Elites.

    Args:
        genotypes: a PyTree containing all the genotypes in the repertoire ordered
            by the centroids. Each leaf has a shape (num_centroids, num_features). The
            PyTree can be a simple JAX array or a more complex nested structure such
            as to represent parameters of neural network in Flax.
        fitnesses: an array that contains the fitness of solutions in each cell of the
            repertoire, ordered by centroids. The array shape is (num_centroids,).
        descriptors: an array that contains the descriptors of solutions in each cell
            of the repertoire, ordered by centroids. The array shape
            is (num_centroids, num_descriptors).
        centroids: an array that contains the centroids of the tessellation. The array
            shape is (num_centroids, num_descriptors).
        extra_scores: extra scores resulting from the evaluation of the genotypes
        keys_extra_scores: keys of the extra scores to store in the repertoire
    """

    descriptors: Descriptor
    centroids: Centroid

    def select(
        self,
        key: RNGKey,
        num_samples: int,
        selector: Optional[Selector[MapElitesRepertoireT]] = None,
    ) -> MapElitesRepertoireT:
        if selector is None:
            selector = UniformSelector(select_with_replacement=True)
        repertoire = selector.select(self, key, num_samples)
        return repertoire

    def add(  # type: ignore
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
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
            batch_of_extra_scores: tree that contains the extra_scores of
                aforementioned genotypes.

        Returns:
            The updated MAP-Elites repertoire.
        """
        if batch_of_extra_scores is None:
            batch_of_extra_scores = {}

        filtered_batch_of_extra_scores = self.filter_extra_scores(batch_of_extra_scores)

        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)

        num_centroids = self.centroids.shape[0]
        batch_of_fitnesses = jnp.reshape(
            batch_of_fitnesses, (batch_of_descriptors.shape[0], 1)
        )

        # get fitness segment max
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices.astype(jnp.int32).squeeze(axis=-1),
            num_segments=num_centroids,
        )

        cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0)

        # put dominated fitness to -jnp.inf
        batch_of_fitnesses = jnp.where(
            batch_of_fitnesses == cond_values, batch_of_fitnesses, -jnp.inf
        )

        # get addition condition
        current_fitnesses = jnp.take_along_axis(self.fitnesses, batch_of_indices, 0)
        addition_condition = batch_of_fitnesses > current_fitnesses

        # assign fake position when relevant : num_centroids is out of bound
        batch_of_indices = jnp.where(
            addition_condition, batch_of_indices, num_centroids
        )

        # create new repertoire
        new_repertoire_genotypes = jax.tree.map(
            lambda repertoire_genotypes, new_genotypes: repertoire_genotypes.at[
                batch_of_indices.squeeze(axis=-1)
            ].set(new_genotypes),
            self.genotypes,
            batch_of_genotypes,
        )

        # compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_fitnesses
        )
        new_descriptors = self.descriptors.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_descriptors
        )

        # update extra scores
        new_extra_scores = jax.tree.map(
            lambda repertoire_scores, new_scores: repertoire_scores.at[
                batch_of_indices.squeeze(axis=-1)
            ].set(new_scores),
            self.extra_scores,
            filtered_batch_of_extra_scores,
        )

        return MapElitesRepertoire(
            genotypes=new_repertoire_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            centroids=self.centroids,
            extra_scores=new_extra_scores,
            keys_extra_scores=self.keys_extra_scores,
        )

    @classmethod
    def init(  # type: ignore
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        *args,
        extra_scores: Optional[ExtraScores] = None,
        keys_extra_scores: Tuple[str, ...] = (),
        **kwargs,
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
            centroids: tessellation centroids of shape (batch_size, num_descriptors)
            extra_scores: extra scores of the initial genotypes
            keys_extra_scores: keys of the extra scores to store in the repertoire

        Returns:
            an initialized MAP-Elite repertoire
        """

        if extra_scores is None:
            extra_scores = {}

        extra_scores = {
            key: value
            for key, value in extra_scores.items()
            if key in keys_extra_scores
        }

        # retrieve one genotype from the population
        first_genotype = jax.tree.map(lambda x: x[0], genotypes)
        first_extra_scores = jax.tree.map(lambda x: x[0], extra_scores)

        # create a repertoire with default values
        repertoire = cls.init_default(
            genotype=first_genotype,
            centroids=centroids,
            one_extra_score=first_extra_scores,
            keys_extra_scores=keys_extra_scores,
        )

        # add initial population to the repertoire
        new_repertoire = repertoire.add(genotypes, descriptors, fitnesses, extra_scores)

        return new_repertoire  # type: ignore

    @classmethod
    def init_default(
        cls,
        genotype: Genotype,
        centroids: Centroid,
        one_extra_score: Optional[ExtraScores] = None,
        keys_extra_scores: Tuple[str, ...] = (),
    ) -> MapElitesRepertoire:
        """Initialize a Map-Elites repertoire with an initial population of
        genotypes. Requires the definition of centroids that can be computed
        with any method such as CVT or Euclidean mapping.

        Note: this function has been kept outside of the object MapElites, so
        it can be called easily called from other modules.

        Args:
            genotype: the typical genotype that will be stored.
            centroids: the centroids of the repertoire
            keys_extra_scores: keys of the extra scores to store in the repertoire

        Returns:
            A repertoire filled with default values.
        """
        if one_extra_score is None:
            one_extra_score = {}

        one_extra_score = {
            key: value
            for key, value in one_extra_score.items()
            if key in keys_extra_scores
        }

        # get number of centroids
        num_centroids = centroids.shape[0]

        # default fitness is -inf
        default_fitnesses = -jnp.inf * jnp.ones(shape=(num_centroids, 1))

        # default genotypes is all 0
        default_genotypes = jax.tree.map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape, dtype=x.dtype),
            genotype,
        )

        # default descriptor is all zeros
        default_descriptors = jnp.zeros_like(centroids)

        # default extra scores is empty dict
        default_extra_scores = jax.tree.map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape, dtype=x.dtype),
            one_extra_score,
        )

        return cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
            extra_scores=default_extra_scores,
            keys_extra_scores=keys_extra_scores,
        )
