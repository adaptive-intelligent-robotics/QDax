"""Core components of the MAP-Elites algorithm."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.core.map_elites import MAPElites
from qdax.custom_types import Centroid, Genotype, Metrics, RNGKey


class DistributedMAPElites(MAPElites):
    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        genotypes: Genotype,
        centroids: Centroid,
        key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics]:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Before the repertoire is initialised, individuals are gathered from all the
        devices.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tessellation centroids of shape (batch_size, num_descriptors)
            key: a random key used for stochastic operations.

        Returns:
            An initialized MAP-Elite repertoire with the initial state of the emitter,
            and a random key.
        """
        # score initial genotypes
        fitnesses, descriptors, extra_scores = self._scoring_function(genotypes, key)

        # gather across all devices
        (
            gathered_genotypes,
            gathered_fitnesses,
            gathered_descriptors,
        ) = jax.tree.map(
            lambda x: jnp.concatenate(jax.lax.all_gather(x, axis_name="p"), axis=0),
            (genotypes, fitnesses, descriptors),
        )

        # init the repertoire
        repertoire = MapElitesRepertoire.init(
            genotypes=gathered_genotypes,
            fitnesses=gathered_fitnesses,
            descriptors=gathered_descriptors,
            centroids=centroids,
        )

        # get initial state of the emitter
        emitter_state = self._emitter.init(
            key=key,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )
        # calculate the initial metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics]:
        """Performs one iteration of the MAP-Elites algorithm.

        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.

        Before the repertoire is updated, individuals are gathered from all the
        devices.

        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            key: a jax PRNG random key

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """
        # generate offsprings with the emitter
        key, subkey = jax.random.split(key)
        genotypes, extra_info = self._emitter.emit(repertoire, emitter_state, subkey)

        # scores the offsprings
        key, subkey = jax.random.split(key)
        fitnesses, descriptors, extra_scores = self._scoring_function(genotypes, subkey)

        # gather across all devices
        (
            gathered_genotypes,
            gathered_fitnesses,
            gathered_descriptors,
        ) = jax.tree.map(
            lambda x: jnp.concatenate(jax.lax.all_gather(x, axis_name="p"), axis=0),
            (genotypes, fitnesses, descriptors),
        )

        # add genotypes in the repertoire
        repertoire = repertoire.add(
            gathered_genotypes, gathered_descriptors, gathered_fitnesses
        )

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores={**extra_scores, **extra_info},
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics

    def get_distributed_init_fn(
        self, centroids: Centroid, devices: List[Any]
    ) -> Callable[
        [Genotype, RNGKey], Tuple[MapElitesRepertoire, Optional[EmitterState]]
    ]:
        """Create a function that init MAP-Elites in a distributed way.

        Args:
            centroids: centroids that structure the repertoire.
            devices: hardware devices.

        Returns:
            A callable function that inits the MAP-Elites algorithm in a distributed
            way.
        """
        return jax.pmap(  # type: ignore
            partial(self.init, centroids=centroids),
            devices=devices,
            axis_name="p",
        )

    def get_distributed_update_fn(
        self, num_iterations: int, devices: List[Any]
    ) -> Callable[
        [MapElitesRepertoire, Optional[EmitterState], RNGKey],
        Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics],
    ]:
        """Create a function that can do a certain number of updates of
        MAP-Elites in a way that is distributed on several devices.

        Args:
            num_iterations: number of iterations to realize.
            devices: hardware devices to distribute on.

        Returns:
            The update function that can be called directly to apply a sequence
            of MAP-Elites updates.
        """

        @jax.jit
        def _scan_update(
            carry: Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey],
            _: Any,
        ) -> Tuple[Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey], Metrics]:
            """Rewrites the update function in a way that makes it compatible with the
            jax.lax.scan primitive."""
            # unwrap the input
            repertoire, emitter_state, key = carry

            # apply one step of update
            key, subkey = jax.random.split(key)
            (
                repertoire,
                emitter_state,
                metrics,
            ) = self.update(
                repertoire,
                emitter_state,
                subkey,
            )

            return (repertoire, emitter_state, key), metrics

        def update_fn(
            repertoire: MapElitesRepertoire,
            emitter_state: Optional[EmitterState],
            key: RNGKey,
        ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics]:
            """Apply num_iterations of update."""
            (
                repertoire,
                emitter_state,
                key,
            ), metrics = jax.lax.scan(
                _scan_update,
                (repertoire, emitter_state, key),
                (),
                length=num_iterations,
            )
            return repertoire, emitter_state, metrics

        return jax.pmap(update_fn, devices=devices, axis_name="p")  # type: ignore
