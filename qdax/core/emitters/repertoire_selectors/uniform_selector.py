from __future__ import annotations

import jax
from jax import numpy as jnp

from qdax.core.emitters.repertoire_selectors.selector import (
    GARepertoireT,
    Selector,
    unfold_repertoire,
)
from qdax.custom_types import RNGKey


class UniformSelector(Selector[GARepertoireT]):
    """
    The uniform selector selects individuals uniformly at random from the repertoire.
    """

    def __init__(self, select_with_replacement: bool = True):
        self.select_with_replacement = select_with_replacement

    def select(
        self,
        repertoire: GARepertoireT,
        key: RNGKey,
        num_samples: int,
    ) -> GARepertoireT:
        """Selects individuals from the repertoire.

        Args:
            repertoire: The repertoire to select from.
            key: The random key to use for the selection.
            num_samples: The number of individuals to select

        Returns:
            A repertoire containing the selected individuals.
        """
        repertoire_unfolded = unfold_repertoire(repertoire)
        # repertoire_unfolded = repertoire

        size_repertoire = repertoire_unfolded.fitnesses.shape[0]

        repertoire_empty = repertoire_unfolded.fitnesses == -jnp.inf
        repertoire_empty = jnp.any(repertoire_unfolded.fitnesses == -jnp.inf, axis=-1)
        p = (1.0 - repertoire_empty) / jnp.sum(1.0 - repertoire_empty)

        indexes = jnp.arange(size_repertoire)
        key, subkey = jax.random.split(key)
        selected_indexes = jax.random.choice(
            subkey,
            indexes,
            shape=(num_samples,),
            p=p,
            replace=self.select_with_replacement,
        )

        selected: GARepertoireT = jax.tree.map(
            lambda x: x[selected_indexes],
            repertoire_unfolded,
        )

        return selected
