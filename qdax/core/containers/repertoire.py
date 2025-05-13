"""This file contains util functions and a class to define
a repertoire, used to store individuals in the MAP-Elites
algorithm as well as several variants."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import flax.struct

from qdax.core.emitters.repertoire_selectors.selector import Selector
from qdax.custom_types import RNGKey


class Repertoire(flax.struct.PyTreeNode, ABC):
    """Abstract class for any repertoire of genotypes.

    We decided not to add the attributes Genotypes even if
    it will be shared by all children classes because we want
    to keep the parent classes explicit and transparent.
    """

    @classmethod
    @abstractmethod
    def init(cls) -> Repertoire:  # noqa: N805
        """Create a repertoire."""
        pass

    @abstractmethod
    def select(
        self,
        key: RNGKey,
        num_samples: int,
        selector: Optional[Selector] = None,
    ) -> Repertoire:
        """Selects individuals from the repertoire.

        Args:
            key: The random key to use for the selection.
            num_samples: The number of individuals to select.
            selector: The selector to use for the selection.

        Returns:
            A repertoire containing the selected individuals.
        """
        pass

    @abstractmethod
    def add(self) -> Repertoire:  # noqa: N805
        """Implements the rule to add new genotypes to a
        repertoire.

        Returns:
            The updated repertoire.
        """
        pass
