"""Core components of the MAP-Elites algorithm."""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, List, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.map_elites import MAPElites
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)


class DistributedMAPElites(MAPElites):
    @partial(jax.jit, static_argnames=("self"))
    def _device_gathering(
        self, x: Tuple[Genotype, Fitness, Descriptor]
    ) -> Tuple[Genotype, Fitness, Descriptor]:
        """Gather data from the different devices to put them back together.

        Args:
            x: data having to be gathered.

        Returns:
            Gathered data
        """
        return jax.tree_util.tree_map(
            lambda y: jnp.concatenate(jax.lax.all_gather(y, axis_name="p"), axis=0),
            x,
        )

    def get_distributed_init(self, centroids: Centroid, devices: List[Any]) -> Callable:
        """TODO

        Args:
            devices:

        Returns:

        """
        return jax.pmap(  # type: ignore
            partial(self.init, centroids=centroids),
            devices=devices,
            axis_name="p",
        )

    def get_distributed_update(self, devices: List[Any]) -> Callable:
        """TODO

        Args:
            devices: _description_

        Returns:
            _description_
        """
        return jax.pmap(self.update, devices=devices, axis_name="p")
