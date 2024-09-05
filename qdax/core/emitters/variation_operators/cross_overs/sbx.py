from typing import Tuple

import jax
from jax import numpy as jnp

from qdax.core.emitters.variation_operators.cross_overs.abstract_cross_over import (
    CrossOver,
)
from qdax.custom_types import Genotype, RNGKey


class SBXCrossOver(CrossOver):
    def __init__(
        self,
        eta: float,
        minval: float,
        maxval: float,
        cross_over_rate: float = 1.0,
        returns_single_genotype: bool = True,
    ):
        super().__init__(cross_over_rate, returns_single_genotype, minval, maxval)
        self._eta = eta

    def _cross_over(
        self, gen_1: Genotype, gen_2: Genotype, random_key: RNGKey
    ) -> Tuple[Genotype, RNGKey]:
        def _crossover_single_subgen_array(
            _subgen_array_1: jnp.ndarray,
            _subgen_array_2: jnp.ndarray,
            _random_key: RNGKey,
        ) -> jnp.ndarray:
            assert self._minval is not None and self._maxval is not None

            normalized_gen_1 = (_subgen_array_1 - self._minval) / (
                self._maxval - self._minval
            )
            normalized_gen_2 = (_subgen_array_2 - self._minval) / (
                self._maxval - self._minval
            )

            y1 = jnp.minimum(normalized_gen_1, normalized_gen_2)
            y2 = jnp.maximum(normalized_gen_1, normalized_gen_2)

            yl = 0.0
            yu = 1.0

            _random_key, _subkey = jax.random.split(_random_key)
            rand = jax.random.uniform(
                key=_random_key, shape=y1.shape, minval=0, maxval=1, dtype=jnp.float32
            )

            beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
            alpha = 2.0 - beta ** -(self._eta + 1.0)

            alpha_rand = rand * alpha

            betaq = jnp.where(
                rand <= 1.0 / alpha,
                (alpha_rand ** (1.0 / (self._eta + 1.0))),
                (1.0 / (2.0 - alpha_rand)) ** (1.0 / (self._eta + 1.0)),
            )

            c1 = 0.5 * (y1 + y2) - 0.5 * (y2 - y1) * betaq

            c1 = jnp.clip(c1, yl, yu)

            c1 = c1 * (self._maxval - self._minval) + self._minval

            return c1

        keys_arrays_tree, random_key = self.get_keys_arrays_tree(gen_1, random_key)
        new_gen = jax.tree_util.tree_map(
            jax.vmap(_crossover_single_subgen_array),
            gen_1,
            gen_2,
            keys_arrays_tree,
        )
        return new_gen, random_key
