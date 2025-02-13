from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Generic, TypeVar

import jax
import numpy as np

from qdax.custom_types import RNGKey

if TYPE_CHECKING:
    from qdax.core.containers.ga_repertoire import GARepertoire
    from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire

GARepertoireT = TypeVar("GARepertoireT", bound="GARepertoire")
MapElitesRepertoireT = TypeVar("MapElitesRepertoireT", bound="MapElitesRepertoire")


def unfold_repertoire(repertoire: GARepertoireT) -> GARepertoireT:
    """Unfolds a repertoire by flattening its base dimensions.

    Args:
        repertoire: The repertoire to unfold

    Returns:
        The unfolded repertoire with shape (prod(base_shape), *remaining_dims)
    """
    base_shape = repertoire.fitnesses.shape[:-1]
    size_unfolded = int(np.prod(base_shape).item())
    unfolded: GARepertoireT = jax.tree.map(
        lambda x: x.reshape(size_unfolded, *x.shape[len(base_shape) :]),
        repertoire,
    )
    return unfolded


class Selector(abc.ABC, Generic[GARepertoireT]):
    """A selector is an object that selects the individuals from a population."""

    @abc.abstractmethod
    def select(
        self,
        repertoire: GARepertoireT,
        key: RNGKey,
        num_samples: int,
    ) -> GARepertoireT:
        """Selects individuals from the repertoire.

        Args:
            repertoire: The repertoire to select from.
            key: The random key to use for the selection.
            num_samples: The number of individuals to select

        Returns:
            A repertoire containing the selected individuals.
        """
        pass
