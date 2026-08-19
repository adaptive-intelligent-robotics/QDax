"""Emitter used by the Multi-Task MAP-Elites algorithm."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.custom_types import ExtraScores, Genotype, RNGKey


class MultiTaskMixingEmitter(MixingEmitter):
    """Mixing emitter that exposes the descriptors of the parents.

    In Multi-task MAP-Elites, the task on which an offspring is evaluated
    is selected with a tournament biased towards the task of its first
    parent. The descriptors stored in a Multi-task MAP-Elites repertoire
    are the task descriptors, hence this emitter returns the descriptors
    of the first parent of each offspring in the extra information
    dictionary, under the key "parent_descriptors".
    """

    def emit(  # type: ignore
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        key: RNGKey,
    ) -> Tuple[Genotype, ExtraScores]:
        """
        Emitter that performs both mutation and variation, and returns the
        descriptors of the first parent of each offspring.

        Params:
            repertoire: the MAP-Elites repertoire to sample from
            emitter_state: void
            key: a jax PRNG random key

        Returns:
            a batch of offsprings and the descriptors of their first parents
        """
        n_variation = int(self._batch_size * self._variation_percentage)
        n_mutation = self._batch_size - n_variation

        if n_variation > 0:
            sample_key_1, sample_key_2, variation_key = jax.random.split(key, 3)
            parents_1 = repertoire.select(
                sample_key_1, n_variation, selector=self._selector
            )
            parents_2 = repertoire.select(
                sample_key_2, n_variation, selector=self._selector
            )
            x_variation = self._variation_fn(
                parents_1.genotypes, parents_2.genotypes, variation_key
            )
            descriptors_variation = parents_1.descriptors

        if n_mutation > 0:
            sample_key, mutation_key = jax.random.split(key)
            parents = repertoire.select(sample_key, n_mutation, selector=self._selector)
            x_mutation = self._mutation_fn(parents.genotypes, mutation_key)
            descriptors_mutation = parents.descriptors

        if n_variation == 0:
            genotypes = x_mutation
            parent_descriptors = descriptors_mutation
        elif n_mutation == 0:
            genotypes = x_variation
            parent_descriptors = descriptors_variation
        else:
            genotypes = jax.tree.map(
                lambda x_1, x_2: jnp.concatenate([x_1, x_2], axis=0),
                x_variation,
                x_mutation,
            )
            parent_descriptors = jnp.concatenate(
                [descriptors_variation, descriptors_mutation], axis=0
            )

        return genotypes, {"parent_descriptors": parent_descriptors}
