from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Optional, Tuple

import jax

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.cma_emitter import CMAEmitterState
from qdax.core.emitters.cma_pool_emitter import CMAPoolEmitter, CMAPoolEmitterState
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype

from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from jax.tree_util import tree_map
from qdax.core.emitters.emitter import EmitterState
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
)

try:
    from evosax import Strategies
except:
    import warnings

    warnings.warn("evosax not installed, custom CMA_ME will not work")

from qdax.core.emitters.termination import cma_criterion
from qdax.utils.evosax_interface import QDaxReshaper

from qdax.core.emitters.evosax_cma_me import (
    EvosaxCMAMEEmitter,
    EvosaxCMAEmitterState,
    EvosaxCMAImprovementEmitter,
    EvosaxCMAOptimizingEmitter,
    EvosaxCMARndEmitter,
    EvosaxCMARndEmitterState,
)


def net_shape(net):
    return jax.tree_map(lambda x: x.shape, net)


class CMAMEPolicies(EvosaxCMAMEEmitter):
    def __init__(
        self,
        batch_size: int,
        centroids: Centroid,
        min_count: Optional[int] = None,
        max_count: Optional[float] = None,
        es_params=None,
        es_type="Sep_CMA_ES",
    ):
        """
        Class for the emitter of CMA ME from "Covariance Matrix Adaptation for the
        Rapid Illumination of Behavior Space" by Fontaine et al.

        This implementation relies on the Evosax library for ES and adds a wrapper to optimize
        QDax neural networks.

        Args:
            batch_size: number of solutions sampled at each iteration
            genotype_dim: dimension of the genotype space.
            centroids: centroids used for the repertoire.
            min_count: minimum number of CMAES opt step before being considered for
                reinitialisation.
            max_count: maximum number of CMAES opt step authorized.
            es_params: parameters of the ES algorithm.
            es_type: type of ES algorithm from Evosax, default for policies is Separable CMA-ES.
        """
        assert Strategies
        self._batch_size = batch_size
        self.es_params = es_params
        self.es_type = es_type

        # Delay until we have genomes
        self.es = None

        # minimum number of emitted solution before an emitter can be re-initialized
        if min_count is None:
            min_count = 0

        self._min_count = min_count

        if max_count is None:
            max_count = jnp.inf

        self._max_count = max_count

        self._centroids = centroids

        if es_type == "CMA_ES":
            self.stop_condition = cma_criterion
        else:
            self.stop_condition = lambda f, s, p: False

        self.reshaper = None

    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[CMAEmitterState, RNGKey]:
        """
        Initializes the CMA-ME emitter


        Args:
            init_genotypes: initial genotypes to add to the grid.
            random_key: a random key to handle stochastic operations.

        Returns:
            The initial state of the emitter.
        """
        # Initialisation requires one initial genotype
        if jax.tree_util.tree_leaves(init_genotypes)[0].shape[0] > 1:
            init_genotypes = jax.tree_util.tree_map(
                lambda x: x[0],
                init_genotypes,
            )

        self.reshaper = QDaxReshaper.init(init_genotypes)

        self.es = Strategies[self.es_type](
            num_dims=self.reshaper.genotype_dim,
            popsize=self.batch_size,
            **self.es_params,
        )

        # Initialize repertoire with default values
        num_centroids = self._centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)

        # Initialize the ES state
        random_key, init_key = jax.random.split(random_key)
        es_params = self.es.default_params
        es_state = self.es.initialize(init_key, params=es_params)

        # return the initial state
        random_key, subkey = jax.random.split(random_key)
        return (
            EvosaxCMAEmitterState(
                random_key=subkey,
                es_state=es_state,
                es_params=es_params,
                previous_fitnesses=default_fitnesses,
                emit_count=0,
            ),
            random_key,
        )

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Optional[MapElitesRepertoire],
        emitter_state: CMAEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        Emits new individuals. Interestingly, this method does not directly modifies
        individuals from the repertoire but sample from a distribution. Hence the
        repertoire is not used in the emit function.

        Args:
            repertoire: a repertoire of genotypes (unused).
            emitter_state: the state of the CMA-ME emitter.
            random_key: a random key to handle random operations.

        Returns:
            New genotypes and a new random key.
        """
        # emit from ES
        es_state = emitter_state.es_state
        es_params = emitter_state.es_params

        random_key, subkey = jax.random.split(random_key)
        offsprings, es_state = self.es.ask(subkey, es_state, es_params)

        # reshape the offsprings
        offsprings = jax.vmap(self.reshaper.unflatten)(offsprings)

        return offsprings, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: CMAEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Optional[EmitterState]:
        """
        Updates the ES-ME emitter state.

        Note: we use the update_state function from CMAES, a function that assumes
        that the candidates are already sorted. We do this because we have to sort
        them in this function anyway, in order to apply the right weights to the
        terms when update theta.

        Args:
            emitter_state: current emitter state
            repertoire: the current genotypes repertoire
            genotypes: the genotypes of the batch of emitted offspring (unused).
            fitnesses: the fitnesses of the batch of emitted offspring.
            descriptors: the descriptors of the emitted offspring.
            extra_scores: unused

        Returns:
            The updated emitter state.
        """

        flat_genotypes = jax.vmap(self.reshaper.flatten)(genotypes)

        return super().state_update(
            emitter_state,
            repertoire,
            flat_genotypes,
            fitnesses,
            descriptors,
            extra_scores,
        )

    def _update_and_init_emitter_state(
        self,
        emitter_state: EvosaxCMAEmitterState,
        repertoire: MapElitesRepertoire,
        emit_count: int,
        random_key: RNGKey,
    ) -> Tuple[EvosaxCMAEmitterState, RNGKey]:
        """Update the emitter state in the case of a reinit event.
        Reinit the cmaes state and use an individual from the repertoire
        as the starting mean.

        Args:
            emitter_state: current cmame state
            repertoire: most recent repertoire
            emit_count: counter of the emitter
            random_key: key to handle stochastic events

        Returns:
            The updated emitter state.
        """

        # re-sample
        random_genotype, random_key = repertoire.sample(random_key, 1)
        # flatten
        random_genotype = jax.vmap(self.reshaper.flatten)(random_genotype)

        # remove the batch dim
        new_mean = jax.tree_util.tree_map(lambda x: x.squeeze(0), random_genotype)

        es_state = emitter_state.es_state.replace(
            mean=new_mean,
        )

        emitter_state = emitter_state.replace(es_state=es_state, emit_count=0)

        return emitter_state, random_key


class PolicyCMAPoolEmitter(CMAPoolEmitter):
    """CMA-ME pool emitter for policies"""

    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[CMAPoolEmitterState, RNGKey]:
        """
        Initializes the CMA-ME emitter


        Args:
            init_genotypes: initial genotypes to add to the grid.
            random_key: a random key to handle stochastic operations.

        Returns:
            The initial state of the emitter.
        """

        emitter_states = []
        for _ in range(self._num_states):
            emitter_state, random_key = self._emitter.init(init_genotypes, random_key)
            emitter_states.append(emitter_state)

        emitter_states = tree_map(lambda *args: jnp.stack(args), *emitter_states)

        # define the emitter state of the pool
        emitter_state = CMAPoolEmitterState(
            current_index=0, emitter_states=emitter_states
        )

        return (
            emitter_state,
            random_key,
        )


class PolicyCMAOptimizingEmitter(CMAMEPolicies, EvosaxCMAOptimizingEmitter):
    """CMA-ME optimizing emitter for policies"""

    pass


class PolicyCMAImprovementEmitter(CMAMEPolicies, EvosaxCMAImprovementEmitter):
    """CMA-ME improvement emitter for policies"""

    pass


class PolicyCMARndEmitter(CMAMEPolicies, EvosaxCMARndEmitter):
    """CMA-ME RND emitter for policies"""

    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[CMAEmitterState, RNGKey]:
        """
        Initializes the RND CMA-ME emitter


        Args:
            init_genotypes: initial genotypes to add to the grid.
            random_key: a random key to handle stochastic operations.

        Returns:
            The initial state of the emitter.
        """
        # Initialisation requires one initial genotype
        if jax.tree_util.tree_leaves(init_genotypes)[0].shape[0] > 1:
            init_genotypes = jax.tree_util.tree_map(
                lambda x: x[0],
                init_genotypes,
            )

        self.reshaper = QDaxReshaper.init(init_genotypes)

        self.es = Strategies[self.es_type](
            num_dims=self.reshaper.genotype_dim,
            popsize=self.batch_size,
            **self.es_params,
        )

        # Initialize repertoire with default values
        num_centroids = self._centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)

        # Initialize the ES state
        random_key, init_key = jax.random.split(random_key)
        es_params = self.es.default_params
        es_state = self.es.initialize(init_key, params=es_params)

        # take a random direction
        random_key, subkey = jax.random.split(random_key)
        random_direction = jax.random.uniform(
            subkey,
            shape=(self._centroids.shape[-1],),
        )

        # return the initial state
        random_key, subkey = jax.random.split(random_key)
        return (
            EvosaxCMARndEmitterState(
                random_key=subkey,
                es_state=es_state,
                es_params=es_params,
                previous_fitnesses=default_fitnesses,
                emit_count=0,
                random_direction=random_direction,
            ),
            random_key,
        )


CMAME_POLICY_CLASSES = {
    "cmame": PolicyCMAOptimizingEmitter,  # default
    "cmame_opt": PolicyCMAOptimizingEmitter,
    "cmame_rnd": PolicyCMARndEmitter,
    "cmame_imp": PolicyCMAImprovementEmitter,
}
