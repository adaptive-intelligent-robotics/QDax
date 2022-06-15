from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Tuple

import flax
import jax
import jax.numpy as jnp
from qdax.types import Centroid, Descriptor, Genotype, RNGKey, Scores
from qdax.utils.pareto_front import (sample_in_masked_pareto_front,
                                     update_masked_pareto_front)


@flax.struct.dataclass
class MOMERepertoire:
    """Class for the grid in Map Elites"""

    grid: Grid
    grid_scores: Scores
    grid_descriptors: Descriptor
    centroids: Centroid

    def save(self, path: str = "./") -> None:
        """
        Save the grid on disk in the form of .npy files.
        """
        jnp.save(path + "grid.npy", self.grid)
        jnp.save(path + "grid_scores.npy", self.grid_scores)
        jnp.save(path + "grid_descriptors.npy", self.grid_descriptors)
        jnp.save(path + "centroids.npy", self.centroids)

    @property
    def grid_size(self) -> int:
        """
        Returns the maximum number of solutions the grid can contain which corresponds
        to the number of cells times the maximum pareto front length.
        """
        return int(self.grid.shape[1] * self.grid.shape[2])

    def sample(
        self, random_key: RNGKey, num_samples: int
    ) -> Tuple[jnp.ndarray, RNGKey]:
        # grid_x of shape (num_centroids, pareto_front,length, genotype_dim)
        # grid_empty of shape (num_centroids, pareto_front_length),
        # vector of boolean True if cell empty False otherwise
        """
        Sample elements in the grid.
        """
        random_key, sub_key1 = jax.random.split(random_key, num=2)
        grid_empty = jnp.any(self.grid_scores == -jnp.inf, axis=-1)
        p = jnp.any(~grid_empty, axis=-1) / jnp.sum(jnp.any(~grid_empty, axis=-1))
        indices = jnp.arange(start=0, stop=grid_empty.shape[0])
        cells_idx = jax.random.choice(sub_key1, indices, shape=(num_samples,), p=p)
        sample_in_fronts = partial(sample_in_masked_pareto_front, num_samples=1)
        random_key = jax.random.split(random_key, num=num_samples)
        sample_in_fronts = jax.vmap(sample_in_fronts)
        elements, random_key = sample_in_fronts(
            self.grid[cells_idx], grid_empty[cells_idx], random_key=random_key
        )
        return elements[:, 0, :], random_key[0, :]

    def add(
        self,
        genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Scores,
    ) -> MOMERepertoire:
        """
        Insert a batch of elements in the grid.
        """
        # grid_fitness of shape (num_centroids, pf_max_size, num_criteria)
        # grid_x of shape (num_centroids, pf_max_size, genotype_dim)
        # grid_empty of shape (num_centroids, pf_max_size),
        # vector of booleans True if cell empty False otherwise
        # x of shape (batch_size, num_criteria)
        # batch_of_descriptors of shape (batch_size, num_descriptors)
        # batch_of_fitnesses of shape (batch_size, num_criteria)

        desc_dim = batch_of_descriptors.shape[1]
        gen_dim = genotypes.shape[1]
        fitness_dim = batch_of_fitnesses.shape[1]

        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)
        concat_array = jnp.concatenate(
            [genotypes, batch_of_descriptors, batch_of_fitnesses, batch_of_indices],
            axis=-1,
        )

        def _add_one(
            carry: MOMERepertoire, x: jnp.ndarray
        ) -> Tuple[MOMERepertoire, Any]:

            genotype, descriptors, fitness, index = jnp.split(
                jnp.ravel(x),
                [gen_dim, gen_dim + desc_dim, gen_dim + desc_dim + fitness_dim],
            )
            index = index.astype(jnp.int32)
            cell, cell_fitness = carry.grid[index], carry.grid_scores[index]
            cell_descriptors = carry.grid_descriptors[index]
            cell_mask = jnp.any(cell_fitness == -jnp.inf, axis=-1)
            cell_fitness, cell, cell_mask = update_masked_pareto_front(
                cell_fitness[0, :],
                jnp.concatenate([cell[0, :], cell_descriptors[0, :]], axis=-1),
                cell_mask[0, :],
                jnp.expand_dims(fitness, axis=0),
                jnp.expand_dims(
                    jnp.concatenate([genotype, descriptors], axis=-1), axis=0
                ),
                jnp.zeros(shape=(1,), dtype=bool),
            )
            cell_desc = cell[:, gen_dim:]
            cell = cell[:, :gen_dim]
            cell_fitness = cell_fitness - jnp.inf * jnp.expand_dims(cell_mask, axis=-1)
            new_grid = carry.grid.at[index].set(cell)
            new_grid_scores = carry.grid_scores.at[index].set(cell_fitness)
            new_grid_descriptors = carry.grid_descriptors.at[index].set(cell_desc)
            carry = carry.replace(  # type: ignore
                grid=new_grid,
                grid_descriptors=new_grid_descriptors,
                grid_scores=new_grid_scores,
            )
            return carry, ()

        self, _ = jax.lax.scan(_add_one, self, concat_array)

        return self
