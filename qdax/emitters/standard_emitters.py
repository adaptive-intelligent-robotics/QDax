from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from qdax.algorithms.map_elites import MapElitesRepertoire
from qdax.emitters.emitter import Emitter
from qdax.types import EmitterState, Genotype, RNGKey


class MixingEmitter(Emitter):
    def __init__(
        self,
        mutation_fn: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        crossover_fn: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        crossover_percentage: float,
        batch_size: int,
    ) -> None:
        self._mutation_fn = mutation_fn
        self._crossover_fn = crossover_fn
        self._crossover_percentage = crossover_percentage
        self._batch_size = batch_size

    @partial(
        jax.jit,
        static_argnames=("self"),
    )
    def emit_fn(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: EmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, EmitterState, RNGKey]:
        """
        Emitter that performs both mutation and crossover. Two batches of
        crossover_percentage * batch_size genotypes are sampled in the repertoire,
        copied and cross-over to obtain new offsprings. One batch of
        (1.0 - crossover_percentage) * batch_size genotypes are sampled in the
        repertoire, copied and mutated.

        Note: this emitter has no state. A fake none state must be added
        through a function redefinition to make this emitter usable with MAP-Elites.

        Params:
            repertoire: the MAP-Elites repertoire to sample from
            emitter_state: void
            random_key: a jax PRNG random key

        Returns:
            a batch of offsprings
            emitter_state: void
            a new jax PRNG key
        """
        n_crossover = int(self._batch_size * self._crossover_percentage)
        n_mutation = self._batch_size - n_crossover

        if n_crossover > 0:
            x1, random_key = repertoire.sample(random_key, n_crossover)
            x2, random_key = repertoire.sample(random_key, n_crossover)

            x_crossover, random_key = self._crossover_fn(x1, x2, random_key)

        if n_mutation > 0:
            x1, random_key = repertoire.sample(random_key, n_mutation)
            x_mutation, random_key = self._mutation_fn(x1, random_key)

        if n_crossover == 0:
            genotypes = x_mutation
        elif n_mutation == 0:
            genotypes = x_crossover
        else:
            genotypes = jax.tree_map(
                lambda x_1, x_2: jnp.concatenate([x_1, x_2], axis=0),
                x_crossover,
                x_mutation,
            )

        return genotypes, emitter_state, random_key
