from typing import Tuple

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.core.emitters.selectors.abstract_selector import Selector
from qdax.custom_types import Genotype, RNGKey


class UniformSelector(Selector):
    def select(
        self,
        number_parents_to_select: int,
        repertoire: Repertoire,
        emitter_state: EmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, EmitterState, RNGKey]:
        """
        Uniform selection of parents
        """
        selected_parents, random_key = repertoire.sample(
            random_key, number_parents_to_select
        )
        return selected_parents, emitter_state, random_key
