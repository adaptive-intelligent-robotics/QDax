from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import jax
from jax import numpy as jnp

from qdax.core.emitters.repertoire_selectors.selector import Selector
from qdax.custom_types import Mask, RNGKey

if TYPE_CHECKING:
    from qdax.core.containers.mome_repertoire import MOMERepertoire

MOMERepertoireT = TypeVar("MOMERepertoireT", bound="MOMERepertoire")


class MOMEUniformSelector(Selector[MOMERepertoireT]):
    """
    The default uniform selector for MOMERepertoire.
    It selects cells uniformly at random from the repertoire.
    And then, it selects individuals uniformly at random from the selected cells.
    """

    def _sample_in_masked_pareto_front(
        self,
        pareto_front: MOMERepertoireT,
        mask: Mask,
        key: RNGKey,
    ) -> MOMERepertoireT:
        """Sample one single individual in masked pareto front.

        Args:
            pareto_front: the individuals of a pareto front
            mask: a mask associated to the front
            key: a random key to handle stochastic operations

        Returns:
            A single individual among the pareto front.
        """
        p = (1.0 - mask) / jnp.sum(1.0 - mask)

        size_pareto_front = pareto_front.fitnesses.shape[0]
        indexes = jnp.arange(size_pareto_front)

        key, subkey = jax.random.split(key)
        selected_indexes = jax.random.choice(subkey, indexes, shape=(1,), p=p)

        selected: MOMERepertoireT = jax.tree.map(
            lambda x: x[selected_indexes],
            pareto_front,
        )

        return selected

    def select(
        self,
        repertoire: MOMERepertoireT,
        key: RNGKey,
        num_samples: int,
    ) -> MOMERepertoireT:
        """Select elements in the repertoire.

        This method sample a non-empty pareto front, and then sample
        genotypes from this pareto front.

        Args:
            repertoire: the repertoire to select from.
            key: a random key to handle stochasticity.
            num_samples: number of samples to retrieve from the repertoire.

        Returns:
            A repertoire containing the selected individuals.
        """

        # create sampling probability for the cells
        repertoire_empty = jnp.any(repertoire.fitnesses == -jnp.inf, axis=-1)
        occupied_cells = jnp.any(~repertoire_empty, axis=-1)

        p = occupied_cells / jnp.sum(occupied_cells)

        # possible indices - num cells
        indices = jnp.arange(start=0, stop=repertoire_empty.shape[0])

        # choose idx - among indices of cells that are not empty
        key, subkey = jax.random.split(key)
        cells_idx = jax.random.choice(subkey, indices, shape=(num_samples,), p=p)

        # get genotypes (front) from the chosen indices
        pareto_fronts = jax.tree.map(lambda x: x[cells_idx], repertoire)

        # prepare second sampling function
        sample_in_fronts = jax.vmap(self._sample_in_masked_pareto_front)

        # sample genotypes from the pareto front
        subkeys = jax.random.split(key, num=num_samples)
        selected = sample_in_fronts(  # type: ignore
            pareto_front=pareto_fronts,
            mask=repertoire_empty[cells_idx],
            key=subkeys,
        )

        # remove the dim coming from pareto front
        selected: MOMERepertoireT = jax.tree.map(lambda x: x.squeeze(axis=1), selected)

        return selected
