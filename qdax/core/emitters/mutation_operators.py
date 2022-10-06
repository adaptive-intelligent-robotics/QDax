"""File defining mutation and crossover functions."""

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.types import Genotype, RNGKey


def _polynomial_mutation(
    x: jnp.ndarray,
    random_key: RNGKey,
    proportion_to_mutate: float,
    eta: float,
    minval: float,
    maxval: float,
) -> jnp.ndarray:
    """Base polynomial mutation for one genotype.

    Proportion to mutate between 0 and 1
    Assumed to be of shape (genotype_dim,)

    Args:
        x: parameters.
        random_key: a random key
        proportion_to_mutate: the proportion of the given parameters
            that need to be mutated.
        eta: the inverse of the power of the mutation applied.
        minval: range of the perturbation applied by the mutation.
        maxval: range of the perturbation applied by the mutation.

    Returns:
        New parameters.
    """

    # Select positions to mutate
    num_positions = x.shape[0]
    positions = jnp.arange(start=0, stop=num_positions)
    num_positions_to_mutate = int(proportion_to_mutate * num_positions)
    random_key, subkey = jax.random.split(random_key)
    selected_positions = jax.random.choice(
        key=subkey, a=positions, shape=(num_positions_to_mutate,), replace=False
    )

    # New values
    mutable_x = x[selected_positions]
    delta_1 = (mutable_x - minval) / (maxval - minval)
    delta_2 = (maxval - mutable_x) / (maxval - minval)
    mutpow = 1.0 / (1.0 + eta)

    # Randomly select where to put delta_1 and delta_2
    random_key, subkey = jax.random.split(random_key)
    rand = jax.random.uniform(
        key=subkey,
        shape=delta_1.shape,
        minval=0,
        maxval=1,
        dtype=jnp.float32,
    )

    value1 = 2.0 * rand + (jnp.power(delta_1, 1.0 + eta) * (1.0 - 2.0 * rand))
    value2 = 2.0 * (1 - rand) + 2.0 * (jnp.power(delta_2, 1.0 + eta) * (rand - 0.5))
    value1 = jnp.power(value1, mutpow) - 1.0
    value2 = 1.0 - jnp.power(value2, mutpow)

    delta_q = jnp.zeros_like(mutable_x)
    delta_q = jnp.where(rand < 0.5, value1, delta_q)
    delta_q = jnp.where(rand >= 0.5, value2, delta_q)

    # Mutate values
    x = x.at[selected_positions].set(mutable_x + (delta_q * (maxval - minval)))

    # Back in bounds if necessary (floating point issues)
    x = jnp.clip(x, minval, maxval)

    return x


def polynomial_mutation(
    x: Genotype,
    random_key: RNGKey,
    proportion_to_mutate: float,
    eta: float,
    minval: float,
    maxval: float,
) -> Tuple[Genotype, RNGKey]:
    """
    Polynomial mutation over several genotypes

    Parameters:
        x: array of genotypes to transform (real values only)
        random_key: RNG key for reproducibility.
            Assumed to be of shape (batch_size, genotype_dim)
        proportion_to_mutate (float): proportion of variables to mutate in
            each genotype (must be in [0, 1]).
        eta: scaling parameter, the larger the more spread the new
            values will be.
        minval: minimum value to clip the genotypes.
        maxval: maximum value to clip the genotypes.

    Returns:
        New genotypes - same shape as input and a new RNG key
    """
    random_key, subkey = jax.random.split(random_key)
    batch_size = jax.tree_util.tree_leaves(x)[0].shape[0]
    mutation_key = jax.random.split(subkey, num=batch_size)
    mutation_fn = partial(
        _polynomial_mutation,
        proportion_to_mutate=proportion_to_mutate,
        eta=eta,
        minval=minval,
        maxval=maxval,
    )
    mutation_fn = jax.vmap(mutation_fn)
    x = jax.tree_util.tree_map(lambda x_: mutation_fn(x_, mutation_key), x)
    return x, random_key


def _polynomial_crossover(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    random_key: RNGKey,
    proportion_var_to_change: float,
) -> jnp.ndarray:
    """
    Base crossover for one pair of genotypes.

    x1 and x2 should have the same shape
    In this function we assume x1 shape and x2 shape to be (genotype_dim,)
    """
    num_var_to_change = int(proportion_var_to_change * x1.shape[0])
    indices = jnp.arange(start=0, stop=x1.shape[0])
    selected_indices = jax.random.choice(
        random_key, indices, shape=(num_var_to_change,)
    )
    x = x1.at[selected_indices].set(x2[selected_indices])
    return x


def polynomial_crossover(
    x1: Genotype,
    x2: Genotype,
    random_key: RNGKey,
    proportion_var_to_change: float,
) -> Tuple[Genotype, RNGKey]:
    """
    Crossover over a set of pairs of genotypes.

    Batched version of _simple_crossover_function
    x1 and x2 should have the same shape
    In this function we assume x1 shape and x2 shape to be
    (batch_size, genotype_dim)

    Parameters:
        x1: first batch of genotypes
        x2: second batch of genotypes
        random_key: RNG key for reproducibility
        proportion_var_to_change: proportion of variables to exchange
            between genotypes (must be [0, 1])

    Returns:
        New genotypes and a new RNG key
    """

    random_key, subkey = jax.random.split(random_key)
    batch_size = jax.tree_util.tree_leaves(x2)[0].shape[0]
    crossover_keys = jax.random.split(subkey, num=batch_size)
    crossover_fn = partial(
        _polynomial_crossover,
        proportion_var_to_change=proportion_var_to_change,
    )
    crossover_fn = jax.vmap(crossover_fn)
    # TODO: check that key usage is correct
    x = jax.tree_util.tree_map(
        lambda x1_, x2_: crossover_fn(x1_, x2_, crossover_keys), x1, x2
    )
    return x, random_key


def isoline_variation(
    x1: Genotype,
    x2: Genotype,
    random_key: RNGKey,
    iso_sigma: float,
    line_sigma: float,
    minval: Optional[float] = None,
    maxval: Optional[float] = None,
) -> Tuple[Genotype, RNGKey]:
    """
    Iso+Line-DD Variation Operator [1] over a set of pairs of genotypes

    Parameters:
        x1 (Genotypes): first batch of genotypes
        x2 (Genotypes): second batch of genotypes
        random_key (RNGKey): RNG key for reproducibility
        iso_sigma (float): spread parameter (noise)
        line_sigma (float): line parameter (direction of the new genotype)
        minval (float, Optional): minimum value to clip the genotypes
        maxval (float, Optional): maximum value to clip the genotypes

    Returns:
        x (Genotypes): new genotypes
        random_key (RNGKey): new RNG key

    [1] Vassiliades, Vassilis, and Jean-Baptiste Mouret. "Discovering the elite
    hypervolume by leveraging interspecies correlation." Proceedings of the Genetic and
    Evolutionary Computation Conference. 2018.
    """

    # Computing line_noise
    random_key, key_line_noise = jax.random.split(random_key)
    batch_size = jax.tree_util.tree_leaves(x1)[0].shape[0]
    line_noise = jax.random.normal(key_line_noise, shape=(batch_size,)) * line_sigma

    def _variation_fn(
        x1: jnp.ndarray, x2: jnp.ndarray, random_key: RNGKey
    ) -> jnp.ndarray:
        iso_noise = jax.random.normal(random_key, shape=x1.shape) * iso_sigma
        x = (x1 + iso_noise) + jax.vmap(jnp.multiply)((x2 - x1), line_noise)

        # Back in bounds if necessary (floating point issues)
        if (minval is not None) or (maxval is not None):
            x = jnp.clip(x, minval, maxval)
        return x

    # create a tree with random keys
    nb_leaves = len(jax.tree_util.tree_leaves(x1))
    random_key, subkey = jax.random.split(random_key)
    subkeys = jax.random.split(subkey, num=nb_leaves)
    keys_tree = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(x1), subkeys)

    # apply isolinedd to each branch of the tree
    x = jax.tree_util.tree_map(
        lambda y1, y2, key: _variation_fn(y1, y2, key), x1, x2, keys_tree
    )

    return x, random_key
