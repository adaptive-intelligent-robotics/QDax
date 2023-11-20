from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.emitters.cma_emitter import CMAEmitter, CMAEmitterState

from qdax.core.emitters.cma_opt_emitter import CMAOptimizingEmitter
from qdax.core.emitters.cma_rnd_emitter import CMARndEmitter, CMARndEmitterState
from qdax.core.emitters.cma_improvement_emitter import CMAImprovementEmitter

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)

from qdax.core.emitters.termination import cma_criterion

try:
    from evosax import EvoState, EvoParams, Strategies
except:
    import warnings

    warnings.warn("evosax not installed, custom CMA_ME will not work")


class EvosaxCMAEmitterState(EmitterState):
    """
    Emitter state for the CMA-ME emitter.

    Args:
        random_key: a random key to handle stochastic operations. Used for
            state update only, another key is used to emit. This might be
            subject to refactoring discussions in the future.
        es_state: state of the underlying CMA-ES algorithm
        es_params: parameters of the underlying CMA-ES algorithm
        previous_fitnesses: store last fitnesses of the repertoire. Used to
            compute the improvment.
        emit_count: count the number of emission events.
    """

    random_key: RNGKey
    es_state: EvoState
    es_params: EvoParams
    previous_fitnesses: Fitness
    emit_count: int


class EvosaxCMARndEmitterState(EvosaxCMAEmitterState):
    """
    Emitter state for the CMA-ME RND emitter.
    """

    random_direction: Descriptor


class EvosaxCMAMEEmitter(CMAEmitter):
    def __init__(
        self,
        batch_size: int,
        genotype_dim: int,
        centroids: Centroid,
        min_count: Optional[int] = None,
        max_count: Optional[float] = None,
        es_params={},
        es_type="CMA_ES",
    ):
        """
        Class for the emitter of CMA ME from "Covariance Matrix Adaptation for the
        Rapid Illumination of Behavior Space" by Fontaine et al.

        This implementation relies on the Evosax library for ES but only optimizes vectors (not neural networks).

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
        self._batch_size = batch_size

        print(es_params)

        # define an Evosax ES instance
        self.es = Strategies[es_type](
            num_dims=genotype_dim,
            popsize=batch_size,
            **es_params,
        )

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

    @partial(jax.jit, static_argnames=("self",))
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
        emitter_state: EvosaxCMAEmitterState,
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

        return offsprings, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: EvosaxCMAEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Optional[EmitterState]:
        """
        Updates the CMA-ME emitter state.

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

        # Compute the improvements - needed for re-init condition
        indices = get_cells_indices(descriptors, repertoire.centroids)
        improvements = fitnesses - emitter_state.previous_fitnesses[indices]

        ranking_criteria = self._ranking_criteria(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            improvements=improvements,
        )

        # get the indices
        sorted_indices = jnp.flip(jnp.argsort(ranking_criteria))

        # sort the candidates
        sorted_candidates = jax.tree_util.tree_map(
            lambda x: x[sorted_indices], genotypes
        )
        sorted_improvements = improvements[sorted_indices]

        # compute reinitialize condition
        emit_count = emitter_state.emit_count + 1

        # check if the criteria are too similar
        sorted_criteria = ranking_criteria[sorted_indices]
        flat_criteria_condition = (
            jnp.linalg.norm(sorted_criteria[0] - sorted_criteria[-1]) < 1e-12
        )

        # check all conditions
        reinitialize = (
            jnp.all(improvements < 0) * (emit_count > self._min_count)
            + (emit_count > self._max_count)
            + self.stop_condition(None, emitter_state.es_state, emitter_state.es_params)
            + flat_criteria_condition
        )

        # If true, draw randomly and re-initialize parameters
        def update_and_reinit(
            operand: Tuple[CMAEmitterState, MapElitesRepertoire, int, RNGKey],
        ) -> Tuple[CMAEmitterState, RNGKey]:
            return self._update_and_init_emitter_state(*operand)

        def update_wo_reinit(
            operand: Tuple[CMAEmitterState, MapElitesRepertoire, int, RNGKey],
        ) -> Tuple[CMAEmitterState, RNGKey]:
            """Update the emitter when no reinit event happened.
            The QDax implementation with custom CMA-ES bypasses the masked update
            of the CMAES, so we remove it too too.
            """

            (emitter_state, repertoire, emit_count, random_key) = operand

            es_state = emitter_state.es_state
            # Update CMA Parameters

            # Flip the sign of the improvements
            flipped_sorted_improvements = -sorted_improvements

            es_state = self.es.tell(
                sorted_candidates,
                flipped_sorted_improvements,
                emitter_state.es_state,
                emitter_state.es_params,
            )

            emitter_state = emitter_state.replace(
                es_state=es_state,
                emit_count=emit_count,
            )

            return emitter_state, random_key

        # Update CMA Parameters
        emitter_state, random_key = jax.lax.cond(
            reinitialize,
            update_and_reinit,
            update_wo_reinit,
            operand=(
                emitter_state,
                repertoire,
                emit_count,
                emitter_state.random_key,
            ),
        )

        # update the emitter state
        emitter_state = emitter_state.replace(
            random_key=random_key, previous_fitnesses=repertoire.fitnesses
        )

        return emitter_state

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

        # remove the batch dim
        new_mean = jax.tree_util.tree_map(lambda x: x.squeeze(0), random_genotype)

        es_state = emitter_state.es_state.replace(
            mean=new_mean,
        )

        emitter_state = emitter_state.replace(es_state=es_state, emit_count=0)

        return emitter_state, random_key


class EvosaxCMAOptimizingEmitter(EvosaxCMAMEEmitter, CMAOptimizingEmitter):
    """CMA-ME Optimizing Emitter using Evosax"""

    pass


class EvosaxCMAImprovementEmitter(EvosaxCMAMEEmitter, CMAImprovementEmitter):
    """CMA-ME Improvement Emitter using Evosax"""

    pass


class EvosaxCMARndEmitter(EvosaxCMAMEEmitter, CMARndEmitter):
    """CMA-ME RND Emitter using Evosax"""

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[CMARndEmitterState, RNGKey]:
        """
        Initializes the RND CMA-ME emitter

        Args:
            init_genotypes: initial genotypes to add to the grid.
            random_key: a random key to handle stochastic operations.

        Returns:
            The initial state of the emitter.
        """
        # Initialize repertoire with default values
        num_centroids = self._centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)

        # Initialize the ES state
        random_key, init_key = jax.random.split(random_key)
        es_params = self.es.default_params
        es_state = self.es.initialize(init_key, params=es_params)

        # take a random direction
        random_key, direction_key = jax.random.split(random_key)
        random_direction = jax.random.uniform(
            direction_key,
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
