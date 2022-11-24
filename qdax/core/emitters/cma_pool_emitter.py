from __future__ import annotations

from functools import partial
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.cma_emitter import CMAEmitter, CMAEmitterState
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class CMAPoolEmitterState(EmitterState):
    """
    Emitter state for the pool of CMA emitters.

    This is for a pool of homogeneous emitters.

    Args:
        current_index: the index of the current emitter state used.
        emitter_states: the batch of emitter states currently used.
    """

    current_index: int
    emitter_states: CMAEmitterState


class CMAPoolEmitter(Emitter):
    def __init__(self, num_states: int, emitter: CMAEmitter):
        """Instantiate a pool of homogeneous emitters.

        Args:
            num_states: the number of emitters to consider. We can use a
                single emitter object and a batched emitter state.
            emitter: the type of emitter for the pool.
        """
        self._num_states = num_states
        self._emitter = emitter

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._emitter.batch_size

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[CMAPoolEmitterState, RNGKey]:
        """
        Initializes the CMA-MEGA emitter


        Args:
            init_genotypes: initial genotypes to add to the grid.
            random_key: a random key to handle stochastic operations.

        Returns:
            The initial state of the emitter.
        """

        def scan_emitter_init(
            carry: RNGKey, unused: Any
        ) -> Tuple[RNGKey, CMAEmitterState]:
            random_key = carry
            emitter_state, random_key = self._emitter.init(init_genotypes, random_key)
            return random_key, emitter_state

        # init all the emitter states
        random_key, emitter_states = jax.lax.scan(
            scan_emitter_init, random_key, (), length=self._num_states
        )

        # define the emitter state of the pool
        emitter_state = CMAPoolEmitterState(
            current_index=0, emitter_states=emitter_states
        )

        return (
            emitter_state,
            random_key,
        )

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Optional[MapElitesRepertoire],
        emitter_state: CMAPoolEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        Emits new individuals.

        Args:
            repertoire: a repertoire of genotypes (unused).
            emitter_state: the state of the CMA-MEGA emitter.
            random_key: a random key to handle random operations.

        Returns:
            New genotypes and a new random key.
        """

        # retrieve the relevant emitter state
        current_index = emitter_state.current_index
        used_emitter_state = jax.tree_util.tree_map(
            lambda x: x[current_index], emitter_state.emitter_states
        )

        # use it to emit offsprings
        offsprings, random_key = self._emitter.emit(
            repertoire, used_emitter_state, random_key
        )

        return offsprings, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: CMAPoolEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Optional[EmitterState]:
        """
        Updates the emitter state.

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

        # retrieve the emitter that has been used and it's emitter state
        current_index = emitter_state.current_index
        emitter_states = emitter_state.emitter_states

        used_emitter_state = jax.tree_util.tree_map(
            lambda x: x[current_index], emitter_states
        )

        # update the used emitter state
        used_emitter_state = self._emitter.state_update(
            emitter_state=used_emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update the emitter state
        emitter_states = jax.tree_util.tree_map(
            lambda x, y: x.at[current_index].set(y), emitter_states, used_emitter_state
        )

        # determine the next emitter to be used
        emit_counts = emitter_states.emit_count

        new_index = jnp.argmin(emit_counts)

        emitter_state = emitter_state.replace(
            current_index=new_index, emitter_states=emitter_states
        )

        return emitter_state  # type: ignore
