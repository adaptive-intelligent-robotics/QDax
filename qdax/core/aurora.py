"""Core class of the AURORA algorithm."""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import flax.struct
import jax
import jax.numpy as jnp
from chex import ArrayTree

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.containers.unstructured_repertoire import UnstructuredRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.types import Descriptor, Fitness, Genotype, Metrics, Params, RNGKey, Observation


@flax.struct.dataclass
class AuroraExtraInfo:
    model_params: Params
    mean_observations: jnp.ndarray
    std_observations: jnp.ndarray


class AURORA:
    """Core elements of the AURORA algorithm.

    Args:
        scoring_function: a function that takes a batch of genotypes and compute
            their fitnesses and descriptors
        emitter: an emitter is used to suggest offsprings given a MAPELites
            repertoire. It has two compulsory functions. A function that takes
            emits a new population, and a function that update the internal state
            of the emitter.
        metrics_function: a function that takes a repertoire and computes
            any useful metric to track its evolution
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey],
            Tuple[Fitness, Descriptor, ArrayTree, RNGKey],
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
        bd_extraction_fn: Callable[
            [QDTransition, jnp.ndarray, Params, Observation, Observation], Descriptor
        ],
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._bd_extraction_fn = bd_extraction_fn

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        init_genotypes: Genotype,
        random_key: RNGKey,
        model_params: Params,
        mean_observations: jnp.ndarray,
        std_observations: jnp.ndarray,
        l_value: jnp.ndarray,
        max_size: int,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey]:
        """Initialize an unstructured repertoire with an initial population of
        genotypes. Requires the definition of centroids that can be computed with
        any method such as CVT or Euclidean mapping.

        Args:
            init_genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            random_key: a random key used for stochastic operations.
            model_params: parameters of the model used to define the behavior
                descriptors.
            mean_observations: mean of the observations gathered.
            std_observations: standard deviation of the observations
                gathered.

        Returns:
            an initialized unstructured repertoire with the initial state of
            the emitter.
        """
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            init_genotypes,
            random_key,
            model_params,
            mean_observations,
            std_observations,
        )

        repertoire = UnstructuredRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            observations=extra_scores["last_valid_observations"],  # type: ignore
            l_value=l_value,
            max_size=max_size,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
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
        model_params: Params,
        mean_observations: jnp.ndarray,
        std_observations: jnp.ndarray,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics, RNGKey]:
        """Main step of the AURORA algorithm.


        Performs one iteration of the AURORA algorithm.
        1. A batch of genotypes is sampled in the archive and the genotypes are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the archive.

        Args:
            repertoire: unstructured repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key
            model_params: params of the model used to define the behavior descriptor.
            mean_observations: mean of the observations gathered.
            std_observations: standard deviation of the observations
                gathered.

        Results:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new key
        """
        # generate offsprings with the emitter
        genotypes, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )
        # scores the offsprings
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes,
            random_key,
            model_params,
            mean_observations,
            std_observations,
        )

        # add genotypes and observations in the repertoire
        repertoire = repertoire.add(
            genotypes, descriptors, fitnesses, extra_scores["last_valid_observations"]
        )

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics, random_key
