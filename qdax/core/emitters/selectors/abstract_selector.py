import abc
from typing import Tuple

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.custom_types import Genotype, RNGKey


class Selector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def select(
        self,
        number_parents_to_select: int,
        repertoire: Repertoire,
        emitter_state: EmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, EmitterState, RNGKey]: ...
