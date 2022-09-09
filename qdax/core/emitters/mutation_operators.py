"""File defining mutation and crossover functions."""
from typing import Optional, Tuple

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.emitters.selectors.abstract_selector import Selector
from qdax.core.emitters.selectors.uniform import UniformSelector
from qdax.core.emitters.variation_operators.abstract_variation import VariationOperator
from qdax.types import Genotype, RNGKey


class SelectionVariationEmitter(Emitter):
    def __init__(
        self,
        batch_size: int,
        variation_operator: VariationOperator,
        selector: Selector = None,
    ):
        self._batch_size = batch_size
        self._variation_operator = variation_operator

        if selector is not None:
            self._selector = selector
        else:
            self._selector = UniformSelector()

    def emit(
        self,
        repertoire: Optional[Repertoire],
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        number_parents_to_select = (
            self._variation_operator.calculate_number_parents_to_select(
                self._batch_size
            )
        )
        genotypes, emitter_state, random_key = self._selector.select(
            number_parents_to_select, repertoire, emitter_state, random_key
        )
        new_genotypes, random_key = self._variation_operator.apply_with_clip(
            genotypes, emitter_state, random_key
        )
        return new_genotypes, random_key
