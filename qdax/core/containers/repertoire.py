"""This file contains util functions and a class to define
a repertoire, used to store individuals in the MAP-Elites
algorithm as well as several variants."""

from __future__ import annotations

from abc import ABC, abstractclassmethod, abstractmethod

import flax

from qdax.types import Genotype, RNGKey


class Repertoire(flax.struct.PyTreeNode, ABC):
    """Abstract class for any repertoire of genotypes.

    We decided not to add the attributes Genotypes even if
    it will be shared by all children classes because we want
    to keep the parent classes explicit and transparent.
    """

    @abstractclassmethod
    def init(cls) -> Repertoire:  # noqa: N805
        """Create a repertoire."""
        pass

    @abstractmethod
    def sample(
        self,
        random_key: RNGKey,
        num_samples: int,
    ) -> Genotype:
        """Sample genotypes from the repertoire.

        Args:
            random_key: a random key to handle stochasticity.
            num_samples: the number of genotypes to sample.

        Returns:
            The sample of genotypes.
        """
        pass

    @abstractmethod
    def add(self) -> Repertoire:
        """Implements the rule to add new genotypes to a
        repertoire.

        Returns:
            The udpated repertoire.
        """
        pass
