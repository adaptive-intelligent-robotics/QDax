from functools import partial
from typing import Optional, Tuple

import jax
from jax import numpy as jnp

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class MultiEmitterState(EmitterState):
    """State of an emitter than use multiple emitters in a parallel manner.

    WARNING: this is not the emitter state of Multi-Emitter MAP-Elites.

    Args:
        emitter_states: a tuple of emitter states
    """

    emitter_states: Tuple[EmitterState, ...]


class MultiEmitter(Emitter):
    """Emitter that mixes several emitters in parallel.

    WARNING: this is not the emitter of Multi-Emitter MAP-Elites.
    """

    def __init__(
        self,
        emitters: Tuple[Emitter, ...],
    ):
        self.emitters = emitters

    def init(
        self, init_genotypes: Optional[Genotype], random_key: RNGKey
    ) -> Tuple[Optional[EmitterState], RNGKey]:
        """
        Initialize the state of the emitter.

        Args:
            init_genotypes: The genotypes of the initial population.
            random_key: a random key to handle stochastic operations.

        Returns:
            The initial emitter state and a random key.
        """

        # prepare keys for each emitter
        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, len(self.emitters))

        # init all emitter states - gather them
        emitter_states = []
        for emitter, subkey_emitter in zip(self.emitters, subkeys):
            emitter_state, _ = emitter.init(init_genotypes, subkey_emitter)
            emitter_states.append(emitter_state)

        return MultiEmitterState(tuple(emitter_states)), random_key

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Optional[Repertoire],
        emitter_state: Optional[MultiEmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Emit new population. Use all the sub emitters to emit subpopulation
        and gather them.

        Args:
            repertoire: a repertoire of genotypes.
            emitter_state: the current state of the emitter.
            random_key: key for random operations.

        Returns:
            Offsprings and a new random key.
        """
        assert emitter_state is not None
        assert len(emitter_state.emitter_states) == len(self.emitters)

        # prepare subkeys for each sub emitter
        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, len(self.emitters))

        # emit from all emitters and gather offsprings
        all_offsprings = []
        for emitter, sub_emitter_state, subkey_emitter in zip(
            self.emitters,
            emitter_state.emitter_states,
            subkeys,
        ):
            genotype, _ = emitter.emit(repertoire, sub_emitter_state, subkey_emitter)
            batch_size = jax.tree_util.tree_leaves(genotype)[0].shape[0]
            assert batch_size == emitter.batch_size
            all_offsprings.append(genotype)

        # concatenate offsprings together
        offsprings = jax.tree_util.tree_map(
            lambda *x: jnp.concatenate(x, axis=0), *all_offsprings
        )

        return offsprings, random_key

    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: Optional[MultiEmitterState],
        repertoire: Optional[Repertoire] = None,
        genotypes: Optional[Genotype] = None,
        fitnesses: Optional[Fitness] = None,
        descriptors: Optional[Descriptor] = None,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Optional[MultiEmitterState]:
        """Update emitter state by updating all sub emitter states.

        Args:
            emitter_state: current emitter state.
            repertoire: current repertoire of genotypes. Defaults to None.
            genotypes: proposed genotypes. Defaults to None.
            fitnesses: associated fitnesses. Defaults to None.
            descriptors: associated descriptors. Defaults to None.
            extra_scores: associated extra_scores. Defaults to None.

        Returns:
            The updated global emitter state.
        """
        if emitter_state is None:
            return None

        # update all the sub emitter states
        emitter_states = []

        index_start = 0
        for emitter, sub_emitter_state in zip(
            self.emitters,
            emitter_state.emitter_states,
        ):
            index_end = index_start + emitter.batch_size
            sub_gen, sub_fit, sub_desc, sub_extra_scores = jax.tree_util.tree_map(
                lambda x, _index_start=index_start, _index_end=index_end: x[
                    _index_start:_index_end
                ],
                (
                    genotypes,
                    fitnesses,
                    descriptors,
                    extra_scores,
                ),
            )
            index_start = index_end
            new_sub_emitter_state = emitter.state_update(
                sub_emitter_state,
                repertoire,
                sub_gen,
                sub_fit,
                sub_desc,
                sub_extra_scores,
            )
            emitter_states.append(new_sub_emitter_state)

        assert index_start == self.batch_size

        # return the update global emitter state
        return MultiEmitterState(tuple(emitter_states))

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return sum(emitter.batch_size for emitter in self.emitters)
