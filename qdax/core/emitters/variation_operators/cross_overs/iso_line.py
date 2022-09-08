from typing import Optional, Tuple

import jax
from jax import numpy as jnp

from qdax.core.emitters.variation_operators.cross_overs.abstract_cross_over import (
    CrossOver,
)
from qdax.types import Genotype, RNGKey


class IsolineVariationOperator(CrossOver):
    def __init__(
        self,
        iso_sigma: float,
        line_sigma: float,
        cross_over_rate: float = 1.0,
        returns_single_genotype: bool = True,
        minval: Optional[float] = None,
        maxval: Optional[float] = None,
    ):
        super().__init__(
            cross_over_rate=cross_over_rate,
            returns_single_genotype=returns_single_genotype,
            minval=minval,
            maxval=maxval,
        )
        self._iso_sigma = iso_sigma
        self._line_sigma = line_sigma

    def _cross_over(
        self, gen_1: Genotype, gen_2: Genotype, random_key: RNGKey
    ) -> Tuple[Genotype, RNGKey]:
        # Computing line_noise
        random_key, key_line_noise = jax.random.split(random_key)
        batch_size = jax.tree_leaves(gen_1)[0].shape[0]
        line_noise = (
            jax.random.normal(key_line_noise, shape=(batch_size,)) * self._line_sigma
        )

        def _variation_fn(
            _x1: jnp.ndarray, _x2: jnp.ndarray, _random_key: RNGKey
        ) -> jnp.ndarray:
            iso_noise = (
                jax.random.normal(_random_key, shape=_x1.shape) * self._iso_sigma
            )
            x = (_x1 + iso_noise) + jax.vmap(jnp.multiply)((_x2 - _x1), line_noise)

            # Back in bounds if necessary (floating point issues)
            if (self._minval is not None) or (self._maxval is not None):
                x = jnp.clip(x, self._minval, self._maxval)
            return x

        # create a tree with random keys
        keys_tree, random_key = self.get_tree_keys(gen_1, random_key)

        # apply isolinedd to each branch of the tree
        gen_new = jax.tree_map(_variation_fn, gen_1, gen_2, keys_tree)

        return gen_new, random_key
