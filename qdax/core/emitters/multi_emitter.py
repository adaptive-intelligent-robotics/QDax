from typing import Optional, Tuple

import jax
from jax import numpy as jnp

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class MultiEmitterState(EmitterState):
    emitters_state_tuple: Tuple[EmitterState, ...]


class MultiEmitter(Emitter):
    def __init__(self, emitters_tuple: Tuple[Emitter, ...]):
        self.emitters_tuple = emitters_tuple

    def emit(
        self,
        repertoire: Optional[Repertoire],
        multi_emitter_state: Optional[MultiEmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        assert multi_emitter_state is not None
        assert len(multi_emitter_state.emitters_state_tuple) == len(self.emitters_tuple)

        random_key, *_keys_emitters = jax.random.split(
            random_key, len(self.emitters_tuple) + 1
        )

        all_genotypes = []
        for emitter, emitter_state, _key_emitter in zip(
            self.emitters_tuple,
            multi_emitter_state.emitters_state_tuple,
            _keys_emitters,
        ):
            genotype, _ = emitter.emit(repertoire, emitter_state, _key_emitter)
            all_genotypes.append(genotype)

        genotypes_tree = jax.tree_map(
            lambda *x: jnp.concatenate(x, axis=0), *all_genotypes
        )
        return genotypes_tree, random_key

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
            self.emitters_tuple, emitter_state.emitters_state_tuple
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
        random_key, *_keys_emitters_list = jax.random.split(
            random_key, len(self.emitters_tuple) + 1
        )

        list_emitter_states = []
        for emitter, _key_emitter in zip(self.emitters_tuple, _keys_emitters_list):
            emitter_state, _ = emitter.init(init_genotypes, _key_emitter)
            list_emitter_states.append(emitter_state)

        return MultiEmitterState(tuple(list_emitter_states)), random_key
