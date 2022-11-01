from typing import Optional, Tuple

import jax
from jax import numpy as jnp

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class MultiEmitterState(EmitterState):
    emitter_states: Tuple[EmitterState, ...]


class MultiEmitter(Emitter):
    def __init__(self, emitters: Tuple[Emitter, ...]):
        self.emitters = emitters

    def emit(
        self,
        repertoire: Optional[Repertoire],
        emitter_state: Optional[MultiEmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        assert emitter_state is not None
        assert len(emitter_state.emitter_states) == len(self.emitters)

        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, len(self.emitters))

        all_offsprings = []
        for emitter, sub_emitter_state, _key_emitter in zip(
            self.emitters,
            emitter_state.emitter_states,
            subkeys,
        ):
            genotype, _ = emitter.emit(repertoire, sub_emitter_state, _key_emitter)
            all_offsprings.append(genotype)

        offsprings = jax.tree_map(
            lambda *x: jnp.concatenate(x, axis=0), *all_offsprings
        )
        return offsprings, random_key

    def state_update(
        self,
        emitter_state: Optional[MultiEmitterState],
        repertoire: Optional[Repertoire] = None,
        genotypes: Optional[Genotype] = None,
        fitnesses: Optional[Fitness] = None,
        descriptors: Optional[Descriptor] = None,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Optional[MultiEmitterState]:
        if emitter_state is None:
            return None

        list_emitter_states = []

        for emitter, sub_emitter_state in zip(
            self.emitters, emitter_state.emitter_states
        ):
            new_sub_emitter_state = emitter.state_update(
                sub_emitter_state,
                repertoire,
                genotypes,
                fitnesses,
                descriptors,
                extra_scores,
            )
            list_emitter_states.append(new_sub_emitter_state)

        return MultiEmitterState(tuple(list_emitter_states))

    def init(
        self, init_genotypes: Optional[Genotype], random_key: RNGKey
    ) -> Tuple[Optional[EmitterState], RNGKey]:

        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, len(self.emitters))

        list_emitter_states = []
        for emitter, _key_emitter in zip(self.emitters, subkeys):
            emitter_state, _ = emitter.init(init_genotypes, _key_emitter)
            list_emitter_states.append(emitter_state)

        return MultiEmitterState(tuple(list_emitter_states)), random_key
