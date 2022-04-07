from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from qdax.algorithms.map_elites import MapElitesRepertoire
from qdax.types import Genotype, RNGKey


def mixing_emitter(
    repertoire: MapElitesRepertoire,
    random_key: RNGKey,
    mutation_fn: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
    crossover_fn: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
    crossover_percentage: float,
    batch_size: int,
) -> (Genotype, RNGKey):
    """
    Emitter that performs both mutation and crossover. Two batches of
    crossover_percentage * batch_size genotypes are sampled in the repertoire, copied
    and cross-over to obtain new offsprings. One batch of
    (1.0 - crossover_percentage) * batch_size genotypes are sampled in the repertoire,
    copied and mutated.

    Note: this emitter has no state. A fake none state must be added
    through a function redefinition to make this emitter usable with MAP-Elites.

    Params:
        repertoire: the MAP-Elites repertoire to sample from
        random_key: a jax PRNG random key
        mutation_fn: a mutation function
        crossover_fn: a crossover function
        crossover_percentage: percentage of batch of offsprings coming from crossover
        batch_size: size of the batch of new offsprings

    Returns:
        a batch of offsprings
        a new jax PRNG key
    """
    n_crossover = int(batch_size * crossover_percentage)
    n_mutation = batch_size - n_crossover

    if n_crossover > 0:
        x1, random_key = repertoire.sample(random_key, n_crossover)
        x2, random_key = repertoire.sample(random_key, n_crossover)

        x_crossover, random_key = crossover_fn(x1, x2, random_key)

    if n_mutation > 0:
        x1, random_key = repertoire.sample(random_key, n_mutation)
        x_mutation, random_key = mutation_fn(x1, random_key)

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

    return genotypes, random_key
