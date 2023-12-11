"""Core class of the AURORA algorithm."""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import ArrayTree

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.containers.unstructured_repertoire import UnstructuredRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.environments.bd_extractors import AuroraExtraInfo
from qdax.types import (
    Descriptor,
    Fitness,
    Genotype,
    Metrics,
    Observation,
    Params,
    RNGKey,
)


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
        encoder_function: Callable[[Observation, AuroraExtraInfo], Descriptor],
        training_function: Callable[
            [RNGKey, UnstructuredRepertoire, Params, int], AuroraExtraInfo
        ],
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._encoder_fn = encoder_function
        self._train_fn = training_function

    def train(
        self,
        repertoire: UnstructuredRepertoire,
        model_params: Params,
        iteration: int,
        random_key: RNGKey,
    ) -> Tuple[UnstructuredRepertoire, AuroraExtraInfo]:
        random_key, subkey = jax.random.split(random_key)
        aurora_extra_info = self._train_fn(
            random_key,
            repertoire,
            model_params,
            iteration,
        )

        # re-addition of all the new behavioural descriptors with the new ae
        new_descriptors = self._encoder_fn(repertoire.observations, aurora_extra_info)

        return (
            repertoire.init(
                genotypes=repertoire.genotypes,
                fitnesses=repertoire.fitnesses,
                descriptors=new_descriptors,
                observations=repertoire.observations,
                l_value=repertoire.l_value,
                max_size=repertoire.max_size,
            ),
            aurora_extra_info,
        )

    @partial(jax.jit, static_argnames=("self",))
    def container_size_control(
        self,
        repertoire: UnstructuredRepertoire,
        target_size: int,
        previous_error: jnp.ndarray,
    ) -> Tuple[UnstructuredRepertoire, jnp.ndarray]:
        # update the l value
        num_indivs = jnp.sum(repertoire.fitnesses != -jnp.inf)

        # CVC Implementation to keep a constant number of individuals in the archive
        current_error = num_indivs - target_size
        change_rate = current_error - previous_error
        prop_gain = 1 * 10e-6
        l_value = (
            repertoire.l_value + (prop_gain * current_error) + (prop_gain * change_rate)
        )

        repertoire = repertoire.init(
            genotypes=repertoire.genotypes,
            fitnesses=repertoire.fitnesses,
            descriptors=repertoire.descriptors,
            observations=repertoire.observations,
            l_value=l_value,
            max_size=repertoire.max_size,
        )

        return repertoire, current_error

    def init(
        self,
        init_genotypes: Genotype,
        aurora_extra_info: AuroraExtraInfo,
        l_value: jnp.ndarray,
        max_size: int,
        random_key: RNGKey,
    ) -> Tuple[UnstructuredRepertoire, Optional[EmitterState], AuroraExtraInfo, RNGKey]:
        """Initialize an unstructured repertoire with an initial population of
        genotypes. Also performs the first training of the AURORA encoder.

        Args:
            init_genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            aurora_extra_info: information to perform AURORA encodings,
                such as the encoder parameters
            l_value: threshold distance for the unstructured repertoire
            max_size: maximum size of the repertoire
            random_key: a random key used for stochastic operations.

        Returns:
            an initialized unstructured repertoire, with the initial state of
            the emitter, and the updated information to perform AURORA encodings
        """
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            init_genotypes,
            random_key,
        )

        observations = extra_scores["last_valid_observations"]

        descriptors = self._encoder_fn(observations, aurora_extra_info)

        repertoire = UnstructuredRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            observations=observations,
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

        random_key, subkey = jax.random.split(random_key)
        repertoire, updated_aurora_extra_info = self.train(
            repertoire, aurora_extra_info.model_params, iteration=0, random_key=subkey
        )

        return repertoire, emitter_state, updated_aurora_extra_info, random_key

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
        aurora_extra_info: AuroraExtraInfo,
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
            aurora_extra_info: extra info for computing encodings

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
        )

        observations = extra_scores["last_valid_observations"]

        descriptors = self._encoder_fn(observations, aurora_extra_info)

        # add genotypes and observations in the repertoire
        repertoire = repertoire.add(
            genotypes,
            descriptors,
            fitnesses,
            observations,
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
