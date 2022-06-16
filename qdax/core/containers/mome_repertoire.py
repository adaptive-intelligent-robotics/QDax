from __future__ import annotations

from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.repertoire import MapElitesRepertoire, get_cells_indices
from qdax.types import Centroid, Descriptor, Fitness, Genotype, RNGKey
from qdax.utils.mome_utils import (
    sample_in_masked_pareto_front,
    update_masked_pareto_front,
)


class MOMERepertoire(MapElitesRepertoire):
    """Class for the repertoire in MO Map Elites

    Genotypes can only be jnp.ndarray at the moment.
    """

    @property
    def grid_size(self) -> int:
        """
        Returns the maximum number of solutions the repertoire can
        contain which corresponds to the number of cells times the
        maximum pareto front length.
        """
        first_leaf = jax.tree_leaves(self.genotypes)[0]
        return int(first_leaf.shape[1] * first_leaf.shape[2])

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample(
        self, random_key: RNGKey, num_samples: int
    ) -> Tuple[jnp.ndarray, RNGKey]:
        """
        Sample elements in the repertoire.

        grid_x of shape (num_centroids, pareto_front,length, genotype_dim)
        grid_empty of shape (num_centroids, pareto_front_length),
        vector of boolean True if cell empty False otherwise
        """

        grid_empty = jnp.any(self.fitnesses == -jnp.inf, axis=-1)
        p = jnp.any(~grid_empty, axis=-1) / jnp.sum(jnp.any(~grid_empty, axis=-1))
        indices = jnp.arange(start=0, stop=grid_empty.shape[0])

        # choose idx
        random_key, subkey = jax.random.split(random_key)
        cells_idx = jax.random.choice(subkey, indices, shape=(num_samples,), p=p)

        # sample
        sample_in_fronts = jax.vmap(
            partial(sample_in_masked_pareto_front, num_samples=1)
        )

        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, num=num_samples)

        # get genotypes
        pareto_front_genotypes = jax.tree_map(lambda x: x[cells_idx], self.genotypes)

        sampled_genotypes = sample_in_fronts(  # type: ignore
            pareto_front_genotypes=pareto_front_genotypes,
            mask=grid_empty[cells_idx],
            random_key=subkeys,
        )

        print("My sampled elements : ", sampled_genotypes)
        print("My random keys : ", random_key)

        sampled_genotypes = jax.tree_map(lambda x: x.squeeze(), sampled_genotypes)

        return sampled_genotypes, random_key

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
    ) -> MOMERepertoire:
        """
        Insert a batch of elements in the repertoire.

        grid_fitness of shape (num_centroids, pf_max_size, num_criteria)
        grid_x of shape (num_centroids, pf_max_size, genotype_dim)
        grid_empty of shape (num_centroids, pf_max_size),
        vector of booleans True if cell empty False otherwise
        x of shape (batch_size, num_criteria)
        batch_of_descriptors of shape (batch_size, num_descriptors)
        batch_of_fitnesses of shape (batch_size, num_criteria)
        """

        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)

        def _add_one(
            carry: MOMERepertoire,
            data: Tuple[Genotype, Descriptor, Fitness, jnp.ndarray],
        ) -> Tuple[MOMERepertoire, Any]:
            # unwrap data
            genotype, descriptors, fitness, index = data

            index = index.astype(jnp.int32)

            # get cell data
            cell_genotype = jax.tree_map(lambda x: x[index], carry.genotypes)
            cell_fitness = carry.fitnesses[index]
            cell_descriptor = carry.descriptors[index]
            cell_mask = jnp.any(cell_fitness == -jnp.inf, axis=-1)

            print("Cell fitness shape : ", cell_fitness)
            print("Fitness shape : ", fitness.shape)

            print("Cell genotype shape : ", cell_genotype)
            print("Cell descriptor shape : ", cell_descriptor)

            print("---" * 50)

            # update pareto front
            (
                cell_fitness,
                cell_genotype,
                cell_descriptor,
                cell_mask,
            ) = update_masked_pareto_front(
                pareto_front_fitness=cell_fitness.squeeze(),
                pareto_front_genotypes=cell_genotype.squeeze(),
                pareto_front_descriptors=cell_descriptor.squeeze(),
                mask=cell_mask.squeeze(),
                new_batch_of_criteria=jnp.expand_dims(fitness, axis=0),
                new_batch_of_genotypes=jnp.expand_dims(genotype, axis=0),
                new_batch_of_descriptors=jnp.expand_dims(descriptors, axis=0),
                new_mask=jnp.zeros(shape=(1,), dtype=bool),
            )

            print("carry genotypes : ", carry.genotypes)
            print("cell genotype : ", cell_genotype)

            print("carry fitnesses : ", carry.fitnesses)
            print("cell fitness : ", cell_fitness)

            print("carry descriptors : ", carry.descriptors)
            print("cell desc : ", cell_descriptor)

            # update cell fitness
            cell_fitness = cell_fitness - jnp.inf * jnp.expand_dims(cell_mask, axis=-1)

            # update grid
            new_genotypes = jax.tree_map(
                lambda x, y: x.at[index].set(y), carry.genotypes, cell_genotype
            )

            new_fitnesses = carry.fitnesses.at[index].set(cell_fitness)
            new_descriptors = carry.descriptors.at[index].set(cell_descriptor)
            carry = carry.replace(  # type: ignore
                genotypes=new_genotypes,
                descriptors=new_descriptors,
                fitnesses=new_fitnesses,
            )

            # return new grid
            return carry, ()

        self, _ = jax.lax.scan(
            _add_one,
            self,
            (
                batch_of_genotypes,
                batch_of_descriptors,
                batch_of_fitnesses,
                batch_of_indices,
            ),
        )

        return self

    @classmethod
    def init(  # type: ignore
        cls,
        genotypes: jnp.ndarray,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        pareto_front_max_length: int,
    ) -> MOMERepertoire:
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

        # get dimensions
        num_criteria = fitnesses.shape[1]
        num_descriptors = descriptors.shape[1]
        num_centroids = centroids.shape[0]

        # create default values
        default_fitnesses = -jnp.inf * jnp.ones(
            shape=(num_centroids, pareto_front_max_length, num_criteria)
        )
        default_genotypes = jax.tree_map(
            lambda x: jnp.zeros(
                shape=(
                    num_centroids,
                    pareto_front_max_length,
                )
                + x.shape[1:]
            ),
            genotypes,
        )
        default_descriptors = jnp.zeros(
            shape=(num_centroids, pareto_front_max_length, num_descriptors)
        )

        # create repertoire with default values
        repertoire = MOMERepertoire(  # type: ignore
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
        )

        # add first batch of individuals in the repertoire
        new_repertoire = repertoire.add(genotypes, descriptors, fitnesses)

        return new_repertoire  # type: ignore
