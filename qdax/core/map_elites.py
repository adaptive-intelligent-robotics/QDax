"""Core components of the MAP-Elites algorithm."""
from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)


class MAPElites:
    """Core elements of the MAP-Elites algorithm.

    Note: Although very similar to the GeneticAlgorithm, we decided to keep the
    MAPElites class independant of the GeneticAlgorithm class at the moment to keep
    elements explicit.

    Args:
        scoring_function: a function that takes a batch of genotypes and compute
            their fitnesses and descriptors
        emitter: an emitter is used to suggest offsprings given a MAPELites
            repertoire. It has two compulsory functions. A function that takes
            emits a new population, and a function that update the internal state
            of the emitter.
        metrics_function: a function that takes a MAP-Elites repertoire and compute
            any useful metric to track its evolution
    """

    REPERTOIRE_FOLDER = "repertoire"
    EMITTER_STATE_FOLDER = "emitter_state"

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        init_genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey]:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
            init_genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            random_key: a random key used for stochastic operations.

        Returns:
            An initialized MAP-Elite repertoire with the initial state of the emitter,
            and a random key.
        """
        # score initial genotypes
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            init_genotypes, random_key
        )

        # init the repertoire
        repertoire = MapElitesRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return repertoire, emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics, RNGKey]:
        """
        Performs one iteration of the MAP-Elites algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.


        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """
        # generate offsprings with the emitter
        genotypes, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )
        # scores the offsprings
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        # add genotypes in the repertoire
        repertoire = repertoire.add(genotypes, descriptors, fitnesses)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics, random_key

    @partial(jax.jit, static_argnames=("self",))
    def scan_update(
        self,
        carry: Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey],
        unused: Any,
    ) -> Tuple[Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey], Metrics]:
        """Rewrites the update function in a way that makes it compatible with the
        jax.lax.scan primitive.

        Args:
            carry: a tuple containing the repertoire, the emitter state and a
                random key.
            unused: unused element, necessary to respect jax.lax.scan API.

        Returns:
            The updated repertoire and emitter state, with a new random key and metrics.
        """
        repertoire, emitter_state, random_key = carry
        (repertoire, emitter_state, metrics, random_key,) = self.update(
            repertoire,
            emitter_state,
            random_key,
        )

        return (repertoire, emitter_state, random_key), metrics

    @classmethod
    def save(
        cls,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        path: str = os.curdir,
    ) -> None:
        # Save Repertoire, adding '' at the end of the path to add separator '/'
        # as needed by repertoire.save().
        path_folder_repertoire = os.path.join(path, cls.REPERTOIRE_FOLDER, "")
        Path(path_folder_repertoire).mkdir(parents=True, exist_ok=True)
        repertoire.save(path_folder_repertoire)

        # Save Emitter State, adding '' at the end of the path to add separator '/'
        # just in case.
        path_folder_emitter_state = os.path.join(path, cls.EMITTER_STATE_FOLDER, "")
        Path(path_folder_emitter_state).mkdir(parents=True, exist_ok=True)
        if emitter_state is not None:
            emitter_state.save(path_folder_emitter_state)

    @classmethod
    def load(
        cls,
        repertoire_instance: MapElitesRepertoire,
        emitter_state_instance: Optional[EmitterState],
        path: str = os.curdir,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState]]:
        def _genotype_reconstruction_fn(genotype: jnp.ndarray) -> Genotype:
            genotype_tree_structure = jax.tree_structure(repertoire_instance.genotypes)
            return jax.tree_unflatten(genotype_tree_structure, genotype)

        # Load Repertoire, adding '' at the end of the path to add separator '/'
        # as needed by repertoire.load().
        path_folder_repertoire = os.path.join(path, cls.REPERTOIRE_FOLDER, "")
        repertoire_loaded = repertoire_instance.load(
            _genotype_reconstruction_fn, path_folder_repertoire
        )

        # Load Emitter State
        path_folder_emitter_state = os.path.join(path, cls.EMITTER_STATE_FOLDER)
        if emitter_state_instance is not None:
            emitter_state_loaded = emitter_state_instance.load(
                path_folder_emitter_state
            )
        else:
            emitter_state_loaded = None

        return repertoire_loaded, emitter_state_loaded
