from __future__ import annotations

from abc import ABC, abstractclassmethod, abstractmethod

import flax

from qdax.types import Genotype, RNGKey


class Repertoire(flax.struct.PyTreeNode, ABC):
    @abstractclassmethod
    def init(cls) -> Repertoire:  # noqa: N805
        pass

    @abstractmethod
    def sample(
        self,
        random_key: RNGKey,
        num_samples: int,
    ) -> Genotype:
        pass

    @abstractmethod
    def add(self) -> Repertoire:
        pass
