from typing import Optional, Tuple

import jax

from qdax.core.emitters.variation_operators.mutations.abstract_mutation import Mutation
from qdax.custom_types import Genotype, RNGKey


class NormalMutation(Mutation):
    def __init__(
        self,
        sigma: float,
        mutation_rate: float = 1.0,
        minval: Optional[float] = None,
        maxval: Optional[float] = None,
    ):
        super().__init__(mutation_rate=mutation_rate, minval=minval, maxval=maxval)
        self.sigma = sigma

    def _mutate(self, gen: Genotype, random_key: RNGKey) -> Tuple[Genotype, RNGKey]:
        array_keys, random_key = self.get_tree_keys(gen, random_key)

        def _variation_fn(_gen: Genotype, _key: RNGKey) -> Genotype:
            return _gen + jax.random.normal(key=_key, shape=_gen.shape) * self.sigma

        return jax.tree_map(_variation_fn, gen, array_keys), random_key
