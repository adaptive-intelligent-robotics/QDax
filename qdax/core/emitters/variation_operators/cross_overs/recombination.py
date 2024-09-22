from typing import Tuple

from qdax.core.emitters.variation_operators.cross_overs.abstract_cross_over import (
    CrossOver,
)
from qdax.custom_types import Genotype, RNGKey


class RecombinationCrossOver(CrossOver):
    def _cross_over(
        self, gen_original: Genotype, gen_exchange: Genotype, random_key: RNGKey
    ) -> Tuple[Genotype, RNGKey]:
        # The exchange cross over is a simple exchange of the two genotypes
        # the proportion of the two genotypes that are changed is the
        # same as the cross-over rate the parts which are exchanged are
        # randomly selected in CrossOver
        return gen_exchange, random_key
