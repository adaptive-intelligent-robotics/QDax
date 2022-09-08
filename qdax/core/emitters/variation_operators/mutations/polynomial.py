from typing import Tuple

import jax
from jax import numpy as jnp

from qdax.core.emitters.variation_operators.mutations.abstract_mutation import Mutation
from qdax.types import Genotype, RNGKey


class PolynomialMutation(Mutation):
    def __init__(
        self,
        eta: float,
        minval: float,
        maxval: float,
        mutation_rate: float = 1.0,
    ):
        # for polynomial mutation, minval and maxval must be specified and finite
        super().__init__(mutation_rate=mutation_rate, minval=minval, maxval=maxval)
        self._eta = eta

    def _mutate(self, gen: Genotype, random_key: RNGKey) -> Tuple[Genotype, RNGKey]:
        def _mutate_single_subgen_array(
            _subgen_array: jnp.ndarray, _random_key: RNGKey
        ) -> jnp.ndarray:
            assert self._minval is not None and self._maxval is not None

            delta_1 = (_subgen_array - self._minval) / (self._maxval - self._minval)
            delta_2 = (self._maxval - _subgen_array) / (self._maxval - self._minval)
            mutpow = 1.0 / (1.0 + self._eta)

            # Randomly select where to put delta_1 and delta_2
            _random_key, subkey = jax.random.split(_random_key)
            rand = jax.random.uniform(
                key=subkey,
                shape=delta_1.shape,
                minval=0,
                maxval=1,
                dtype=jnp.float32,
            )

            value1 = 2.0 * rand + (
                jnp.power(delta_1, 1.0 + self._eta) * (1.0 - 2.0 * rand)
            )
            value2 = 2.0 * (1 - rand) + 2.0 * (
                jnp.power(delta_2, 1.0 + self._eta) * (rand - 0.5)
            )
            value1 = jnp.power(value1, mutpow) - 1.0
            value2 = 1.0 - jnp.power(value2, mutpow)

            delta_q = jnp.zeros_like(_subgen_array)
            delta_q = jnp.where(rand < 0.5, value1, delta_q)
            delta_q = jnp.where(rand >= 0.5, value2, delta_q)

            # Mutate values
            new_subgen_array = _subgen_array + delta_q * (self._maxval - self._minval)
            return new_subgen_array

        keys_arrays_tree, random_key = self.get_keys_arrays_tree(gen, random_key)

        new_gen = jax.tree_map(_mutate_single_subgen_array, gen, keys_arrays_tree)

        return new_gen, random_key
