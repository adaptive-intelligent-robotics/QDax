import abc
from typing import Optional, Tuple

import jax
from jax import numpy as jnp

from qdax.core.emitters.emitter import EmitterState
from qdax.core.emitters.variation_operators.abstract_variation import VariationOperator
from qdax.custom_types import Genotype, RNGKey


class CrossOver(VariationOperator, abc.ABC):
    def __init__(
        self,
        cross_over_rate: float = 1.0,
        returns_single_genotype: bool = True,
        minval: Optional[float] = None,
        maxval: Optional[float] = None,
    ):
        super().__init__(minval, maxval)
        self.cross_over_rate = cross_over_rate
        self.returns_single_genotype = returns_single_genotype

    @property
    def number_parents_to_select(self) -> int:
        return 2

    @property
    def number_genotypes_returned(self) -> int:
        if self.returns_single_genotype:
            return 1
        else:
            return 2

    def apply_without_clip(
        self,
        genotypes: Genotype,
        emitter_state: EmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        gen_1, gen_2 = self._divide_genotypes(genotypes)
        selected_indices, random_key = self._get_random_positions_to_change(
            gen_1, self.cross_over_rate, random_key
        )
        subgen_1 = self._get_sub_genotypes(gen_1, selected_positions=selected_indices)
        subgen_2 = self._get_sub_genotypes(gen_2, selected_positions=selected_indices)

        if self.returns_single_genotype:

            new_subgen, random_key = self._cross_over(subgen_1, subgen_2, random_key)
            new_gen = self._set_sub_genotypes(gen_1, selected_indices, new_subgen)
            return new_gen, random_key
        else:
            # Not changing random key here to keep same noise for gen_tilde_1 and
            # gen_tilde_2 (as done in the literature)
            new_subgen_1, _ = self._cross_over(subgen_1, subgen_2, random_key)
            new_subgen_2, random_key = self._cross_over(subgen_2, subgen_1, random_key)

            new_gen_1 = self._set_sub_genotypes(gen_1, selected_indices, new_subgen_1)
            new_gen_2 = self._set_sub_genotypes(gen_2, selected_indices, new_subgen_2)

            new_gen = jax.tree_util.tree_map(
                lambda x_1, x_2: jnp.concatenate([x_1, x_2], axis=0),
                new_gen_1,
                new_gen_2,
            )
            return new_gen, random_key

    @abc.abstractmethod
    def _cross_over(
        self, gen_1: Genotype, gen_2: Genotype, random_key: RNGKey
    ) -> Tuple[Genotype, RNGKey]: ...
