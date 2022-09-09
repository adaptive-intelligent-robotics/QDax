import math
from typing import List, Optional, Tuple

from qdax.core.emitters.emitter import EmitterState
from qdax.core.emitters.variation_operators.abstract_variation import VariationOperator
from qdax.types import Genotype, RNGKey


class ComposerVariations(VariationOperator):
    def __init__(
        self,
        variations_operators_list: List[VariationOperator],
        minval: Optional[float] = None,
        maxval: Optional[float] = None,
    ):
        super().__init__(minval, maxval)
        self.variations_list = variations_operators_list

    @property
    def number_parents_to_select(self) -> int:
        numbers_to_select = map(
            lambda x: x.number_parents_to_select, self.variations_list
        )
        return math.prod(numbers_to_select)

    @property
    def number_genotypes_returned(self) -> int:
        numbers_to_return = map(
            lambda x: x.number_genotypes_returned, self.variations_list
        )
        return math.prod(numbers_to_return)

    def apply_without_clip(
        self, genotypes: Genotype, emitter_state: EmitterState, random_key: RNGKey
    ) -> Tuple[Genotype, RNGKey]:
        for variation in self.variations_list:
            genotypes, random_key = variation.apply_with_clip(
                genotypes, emitter_state, random_key
            )
        return genotypes, random_key
