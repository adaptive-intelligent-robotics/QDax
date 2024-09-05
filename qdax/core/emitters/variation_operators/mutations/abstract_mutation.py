import abc
from typing import Optional, Tuple

from qdax.core.emitters.emitter import EmitterState
from qdax.core.emitters.variation_operators.abstract_variation import VariationOperator
from qdax.custom_types import Genotype, RNGKey


class Mutation(VariationOperator, abc.ABC):
    def __init__(
        self,
        mutation_rate: float,
        minval: Optional[float] = None,
        maxval: Optional[float] = None,
    ):
        super().__init__(minval, maxval)
        self.mutation_rate = mutation_rate

    @property
    def number_parents_to_select(self) -> int:
        return 1

    @property
    def number_genotypes_returned(self) -> int:
        return 1

    def apply_without_clip(
        self,
        genotypes: Genotype,
        emitter_state: EmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        selected_indices, random_key = self._get_random_positions_to_change(
            genotypes, self.mutation_rate, random_key
        )
        selected_gens = self._get_sub_genotypes(
            genotypes, selected_positions=selected_indices
        )
        selected_gens_mutated, random_key = self._mutate(selected_gens, random_key)
        new_genotypes = self._set_sub_genotypes(
            genotypes, selected_indices, selected_gens_mutated
        )
        return new_genotypes, random_key

    @abc.abstractmethod
    def _mutate(self, gen: Genotype, random_key: RNGKey) -> Tuple[Genotype, RNGKey]: ...
