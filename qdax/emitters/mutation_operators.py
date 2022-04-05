"""File defining mutation and crossover functions."""

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.types import Genotype, RNGKey


def _polynomial_mutation_function(
    x: jnp.ndarray,
    random_key: RNGKey,
    proportion_to_mutate: float,
    eta: float,
    minval: float,
    maxval: float,
):
    """
    Base polynomial mutation for one genotype.

    Proportion to mutate between 0 and 1
    Assumed to be of shape (genotype_dim,)
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


def polynomial_mutation_function(
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
        x (Genotypes): array of genotypes to transform (real values only)
        random_key (RNGKey): RNG key for reproducibility.
        Assumed to be of shape (batch_size, genotype_dim)

        proportion_to_mutate (float): proportion of variables to mutate in
            each genotype (must be in [0, 1]).
        eta (float): scaling parameter, the larger the more spread the new
        values will be.
        minval (float): minimum value to clip the genotypes.
        maxval (float): maximum value to clip the genotypes.

    Returns:
        x (Genotypes): new genotypes - same shape as input
        random_key (RNGKey): new RNG key
    """
    random_key, subkey = jax.random.split(random_key)
    batch_size = jax.tree_leaves(x)[0].shape[0]
    mutation_key = jax.random.split(subkey, num=batch_size)
    mutation_fn = partial(
        _polynomial_mutation_function,
        proportion_to_mutate=proportion_to_mutate,
        eta=eta,
        minval=minval,
        maxval=maxval,
    )
    mutation_fn = jax.vmap(mutation_fn)
    x = jax.tree_map(lambda x_: mutation_fn(x_, mutation_key), x)
    return x, random_key


def _polynomial_crossover_function(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    random_key: RNGKey,
    proportion_var_to_change: float,
):
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


def polynomial_crossover_function(
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
        x1 (Genotypes): first batch of genotypes
        x2 (Genotypes): second batch of genotypes
        random_key (RNGKey): RNG key for reproducibility
        proportion_var_to_change (float): proportion of variables to exchange
        between genotypes (must be [0, 1])

    Returns:
        x (Genotypes): new genotypes
        random_key (RNGKey): new RNG key
    """

    random_key, subkey = jax.random.split(random_key)
    crossover_keys = jax.random.split(subkey, num=x2.shape[0])
    crossover_fn = partial(
        _polynomial_crossover_function,
        proportion_var_to_change=proportion_var_to_change,
    )
    crossover_fn = jax.vmap(crossover_fn)
    # TODO: check that key usage is correct
    x = jax.tree_map(lambda x1_, x2_: crossover_fn(x1_, x2_, crossover_keys), x1, x2)
    return x, random_key


def isoline_crossover_function(
    x1: Genotype,
    x2: Genotype,
    random_key: RNGKey,
    iso_sigma: float,
    line_sigma: float,
    minval: Optional[float] = None,
    maxval: Optional[float] = None,
) -> Tuple[Genotype, RNGKey]:
    """
    Iso+Line-DD Crossover Operator [1] over a set of pairs of genotypes
    [1] Vassiliades, Vassilis, and Jean-Baptiste Mouret. "Discovering the elite
    hypervolume by leveraging interspecies correlation." Proceedings of the Genetic and
    Evolutionary Computation Conference. 2018.

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
    """

    key, subkey1, subkey2 = jax.random.split(random_key, num=3)

    def _crossover_fn(x1, x2):
        iso_noise = jax.random.normal(subkey1, shape=x1.shape) * iso_sigma
        line_noise = jax.random.normal(subkey2, shape=x2.shape) * line_sigma
        x = (x1 + iso_noise) + line_noise * (x2 - x1)

        # Back in bounds if necessary (floating point issues)
        if (minval is not None) or (maxval is not None):
            x = jnp.clip(x, minval, maxval)
        return x

    x = jax.tree_map(lambda y1, y2: _crossover_fn(y1, y2), x1, x2)

    return x, key
