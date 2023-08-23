"""This file contains the class to define the repertoire used to
store individuals in the Multi-Objective MAP-Elites algorithm as
well as several variants."""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, Spread


def _dispersion(descriptors: jnp.ndarray) -> jnp.ndarray:
    """Computes dispersion of a batch of num_samples descriptors.

    Args:
        descriptors: (num_samples, num_descriptors) array of descriptors.
    Returns:
        The float dispersion of the descriptors (this is represented as a scalar
        jnp.ndarray).
    """

    # Pairwise distances between the descriptors.
    dists = jnp.linalg.norm(descriptors[:, None] - descriptors, axis=2)

    # Compute dispersion -- this is the mean of the unique pairwise distances.
    #
    # Zero out the duplicate distances since the distance matrix is diagonal.
    # Setting k=1 will also remove entries on the diagonal since they are zero.
    dists = jnp.triu(dists, k=1)

    num_samples = len(descriptors)
    n_pairwise = num_samples * (num_samples - 1) / 2.0

    return jnp.sum(dists) / n_pairwise


def _mode(x: jnp.ndarray) -> jnp.ndarray:
    """Computes mode (most common item) of an array.

    The return type is a scalar ndarray.
    """
    unique_vals, counts = jnp.unique(x, return_counts=True, size=x.size)
    return unique_vals[jnp.argmax(counts)]


class MELSRepertoire(MapElitesRepertoire):
    """Class for the repertoire in MAP-Elites Low-Spread.

    This class inherits from MapElitesRepertoire. In addition to the stored data in
    MapElitesRepertoire (genotypes, fitnesses, descriptors, centroids), this repertoire
    also maintains an array of spreads. We overload the save, load, add, and
    init_default methods of MapElitesRepertoire.

    Refer to Mace 2023 for more info on MAP-Elites Low-Spread:
    https://dl.acm.org/doi/abs/10.1145/3583131.3590433

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
        centroids: an array that contains the centroids of the tessellation. The array
            shape is (num_centroids, num_descriptors).
        spreads: an array that contains the spread of solutions in each cell of the
            repertoire, ordered by centroids. The array shape is (num_centroids,).
    """

    spreads: Spread

    def save(self, path: str = "./") -> None:
        """Saves the repertoire on disk in the form of .npy files.

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
        jnp.save(path + "spreads.npy", self.spreads)

    @classmethod
    def load(cls, reconstruction_fn: Callable, path: str = "./") -> MELSRepertoire:
        """Loads a MAP-Elites Low-Spread Repertoire.

        Args:
            reconstruction_fn: Function to reconstruct a PyTree
                from a flat array.
            path: Path where the data is saved. Defaults to "./".

        Returns:
            A MAP-Elites Low-Spread Repertoire.
        """

        flat_genotypes = jnp.load(path + "genotypes.npy")
        genotypes = jax.vmap(reconstruction_fn)(flat_genotypes)

        fitnesses = jnp.load(path + "fitnesses.npy")
        descriptors = jnp.load(path + "descriptors.npy")
        centroids = jnp.load(path + "centroids.npy")
        spreads = jnp.load(path + "spreads.npy")

        return cls(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            spreads=spreads,
        )

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> MELSRepertoire:
        """
        Add a batch of elements to the repertoire.

        The key difference between this method and the default add() in
        MapElitesRepertoire is that it expects each individual to be evaluated
        `num_samples` times, resulting in `num_samples` fitnesses and
        `num_samples` descriptors per individual.

        If multiple individuals may be added to a single cell, this method will
        arbitrarily pick one -- the exact choice depends on the implementation of
        jax.at[].set(), which can be non-deterministic:
        https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
        We do not currently check if one of the multiple individuals dominates the
        others (dominate means that the individual has both highest fitness and lowest
        spread among the individuals for that cell).

        If `num_samples` is only 1, the spreads will default to 0.

        Args:
            batch_of_genotypes: a batch of genotypes to be added to the repertoire.
                Similarly to the self.genotypes argument, this is a PyTree in which
                the leaves have a shape (batch_size, num_features)
            batch_of_descriptors: an array that contains the descriptors of the
                aforementioned genotypes over all evals. Its shape is
                (batch_size, num_samples, num_descriptors). Note that we "aggregate"
                descriptors by finding the most frequent cell of each individual. Thus,
                the actual descriptors stored in the repertoire are just the coordinates
                of the centroid of the most frequent cell.
            batch_of_fitnesses: an array that contains the fitnesses of the
                aforementioned genotypes over all evals. Its shape is (batch_size,
                num_samples)
            batch_of_extra_scores: unused tree that contains the extra_scores of
                aforementioned genotypes.

        Returns:
            The updated repertoire.
        """
        batch_size, num_samples = batch_of_fitnesses.shape

        # Compute indices/cells of all descriptors.
        batch_of_all_indices = get_cells_indices(
            batch_of_descriptors.reshape(batch_size * num_samples, -1), self.centroids
        ).reshape((batch_size, num_samples))

        # Compute most frequent cell of each solution.
        batch_of_indices = jax.vmap(_mode)(batch_of_all_indices)[:, None]

        # Compute dispersion / spread. The dispersion is set to zero if
        # num_samples is 1.
        batch_of_spreads = jax.lax.cond(
            num_samples == 1,
            lambda desc: jnp.zeros(batch_size),
            lambda desc: jax.vmap(_dispersion)(
                desc.reshape((batch_size, num_samples, -1))
            ),
            batch_of_descriptors,
        )
        batch_of_spreads = jnp.expand_dims(batch_of_spreads, axis=-1)

        # Compute canonical descriptors as the descriptor of the centroid of the most
        # frequent cell. Note that this line redefines the earlier batch_of_descriptors.
        batch_of_descriptors = jnp.take_along_axis(
            self.centroids, batch_of_indices, axis=0
        )

        # Compute canonical fitnesses as the average fitness.
        #
        # Shape: (batch_size, 1)
        batch_of_fitnesses = batch_of_fitnesses.mean(axis=-1, keepdims=True)

        num_centroids = self.centroids.shape[0]

        # get current repertoire fitnesses and spreads
        repertoire_fitnesses = jnp.expand_dims(self.fitnesses, axis=-1)
        current_fitnesses = jnp.take_along_axis(
            repertoire_fitnesses, batch_of_indices, 0
        )

        repertoire_spreads = jnp.expand_dims(self.spreads, axis=-1)
        current_spreads = jnp.take_along_axis(repertoire_spreads, batch_of_indices, 0)

        # get addition condition
        addition_condition_fitness = batch_of_fitnesses > current_fitnesses
        addition_condition_spread = batch_of_spreads <= current_spreads
        addition_condition = jnp.logical_and(
            addition_condition_fitness, addition_condition_spread
        )

        # assign fake position when relevant : num_centroids is out of bound
        batch_of_indices = jnp.where(
            addition_condition, x=batch_of_indices, y=num_centroids
        )

        # create new repertoire
        new_repertoire_genotypes = jax.tree_util.tree_map(
            lambda repertoire_genotypes, new_genotypes: repertoire_genotypes.at[
                batch_of_indices.squeeze(axis=-1)
            ].set(new_genotypes),
            self.genotypes,
            batch_of_genotypes,
        )

        # compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_fitnesses.squeeze(axis=-1)
        )
        new_descriptors = self.descriptors.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_descriptors
        )
        new_spreads = self.spreads.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_spreads.squeeze(axis=-1)
        )

        return MELSRepertoire(
            genotypes=new_repertoire_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            centroids=self.centroids,
            spreads=new_spreads,
        )

    @classmethod
    def init_default(
        cls,
        genotype: Genotype,
        centroids: Centroid,
    ) -> MELSRepertoire:
        """Initialize a MAP-Elites Low-Spread repertoire with an initial population of
        genotypes. Requires the definition of centroids that can be computed with any
        method such as CVT or Euclidean mapping.

        Note: this function has been kept outside of the object MELS, so
        it can be called easily called from other modules.

        Args:
            genotype: the typical genotype that will be stored.
            centroids: the centroids of the repertoire.

        Returns:
            A repertoire filled with default values.
        """

        # get number of centroids
        num_centroids = centroids.shape[0]

        # default fitness is -inf
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)

        # default genotypes is all 0
        default_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape, dtype=x.dtype),
            genotype,
        )

        # default descriptor is all zeros
        default_descriptors = jnp.zeros_like(centroids)

        # default spread is inf so that any spread will be less
        default_spreads = jnp.full(shape=num_centroids, fill_value=jnp.inf)

        return cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
            spreads=default_spreads,
        )
