from functools import partial
from typing import Optional, Tuple

import jax
import numpy as np
from chex import ArrayTree
from jax import numpy as jnp

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


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
        indexes_separation_batches = self.get_indexes_separation_batches(emitters)
        self.indexes_start_batches = indexes_separation_batches[:-1]
        self.indexes_end_batches = indexes_separation_batches[1:]

    @staticmethod
    def get_indexes_separation_batches(
        emitters: Tuple[Emitter, ...]
    ) -> Tuple[int, ...]:
        """Get the indexes of the separation between batches of each emitter.

        Args:
            emitters: the emitters

        Returns:
            a tuple of tuples of indexes
        """
        indexes_separation_batches = np.cumsum(
            [0] + [emitter.batch_size for emitter in emitters]
        )
        return tuple(indexes_separation_batches)

    def init(
        self,
        key: RNGKey,
        repertoire: Repertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> Optional[EmitterState]:
        """
        Initialize the state of the emitter.

        Args:
            genotypes: The genotypes of the initial population.
            key: a random key to handle stochastic operations.

        Returns:
            The initial emitter state.
        """

        # prepare keys for each emitter
        keys = jax.random.split(key, len(self.emitters))

        # init all emitter states - gather them
        emitter_states = []
        for emitter, key_emitter in zip(self.emitters, keys):
            emitter_state = emitter.init(
                key_emitter,
                repertoire,
                genotypes,
                fitnesses,
                descriptors,
                extra_scores,
            )
            emitter_states.append(emitter_state)

        return MultiEmitterState(tuple(emitter_states))

    def emit(
        self,
        repertoire: Optional[Repertoire],
        emitter_state: Optional[MultiEmitterState],
        key: RNGKey,
    ) -> Tuple[Genotype, ExtraScores]:
        """Emit new population. Use all the sub emitters to emit subpopulation
        and gather them.

        Args:
            repertoire: a repertoire of genotypes.
            emitter_state: the current state of the emitter.
            key: key for random operations.

        Returns:
            Offsprings.
        """
        assert emitter_state is not None
        assert len(emitter_state.emitter_states) == len(self.emitters)

        # prepare subkeys for each sub emitter
        keys = jax.random.split(key, len(self.emitters))

        # emit from all emitters and gather offsprings
        all_offsprings = []
        all_extra_info: ExtraScores = {}
        for emitter, sub_emitter_state, key_emitter in zip(
            self.emitters,
            emitter_state.emitter_states,
            keys,
        ):
            genotype, extra_info = emitter.emit(
                repertoire, sub_emitter_state, key_emitter
            )
            batch_size = jax.tree.leaves(genotype)[0].shape[0]
            assert batch_size == emitter.batch_size
            all_offsprings.append(genotype)
            all_extra_info = {**all_extra_info, **extra_info}

        # concatenate offsprings together
        offsprings = jax.tree.map(
            lambda *x: jnp.concatenate(x, axis=0), *all_offsprings
        )
        return offsprings, all_extra_info

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

        def _get_sub_pytree(pytree: ArrayTree, start: int, end: int) -> ArrayTree:
            return jax.tree.map(lambda x: x[start:end], pytree)

        for emitter, sub_emitter_state, index_start, index_end in zip(
            self.emitters,
            emitter_state.emitter_states,
            self.indexes_start_batches,
            self.indexes_end_batches,
        ):
            # update with all genotypes, fitnesses, etc...
            if emitter.use_all_data:
                new_sub_emitter_state = emitter.state_update(
                    sub_emitter_state,
                    repertoire,
                    genotypes,
                    fitnesses,
                    descriptors,
                    extra_scores,
                )
                emitter_states.append(new_sub_emitter_state)
            # update only with the data of the emitted genotypes
            else:
                # extract relevant data
                sub_gen, sub_fit, sub_desc, sub_extra_scores = jax.tree.map(
                    partial(_get_sub_pytree, start=index_start, end=index_end),
                    (
                        genotypes,
                        fitnesses,
                        descriptors,
                        extra_scores,
                    ),
                )
                # update only with the relevant data
                new_sub_emitter_state = emitter.state_update(
                    sub_emitter_state,
                    repertoire,
                    sub_gen,
                    sub_fit,
                    sub_desc,
                    sub_extra_scores,
                )
                emitter_states.append(new_sub_emitter_state)

        # return the update global emitter state
        return MultiEmitterState(tuple(emitter_states))

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return sum(emitter.batch_size for emitter in self.emitters)
