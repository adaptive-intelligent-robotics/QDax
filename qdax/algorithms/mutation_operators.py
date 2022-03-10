from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from qdax.algorithms.types import Genotypes, RNGKey


def _polynomial_mutation_function(
    x: jnp.ndarray,
    random_keys: jnp.ndarray,
    proportion_to_mutate: float,
    eta: float,
    minval: float,
    maxval: float,
):
    """
    Base polynomial mutation for one genotype

    proportion to mutate between 0 and 1
    assumed to be of shape (genotype_dim,)
    """

    # Select positions to mutate
    num_positions = x.shape[0]
    positions = jnp.arange(start=0, stop=num_positions)
    num_positions_to_mutate = int(proportion_to_mutate * num_positions)
    selected_positions = jax.random.choice(
        key=random_keys[0], a=positions, shape=(num_positions_to_mutate,), replace=False
    )

    # New values
    mutable_x = x[selected_positions]
    delta_1 = (mutable_x - minval) / (maxval - minval)
    delta_2 = (maxval - mutable_x) / (maxval - minval)
    mutpow = 1.0 / (1.0 + eta)

    # Randomly select where to put delta_1 and delta_2
    rand = jax.random.uniform(
        key=random_keys[1],
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
    x: Genotypes,
    random_key: RNGKey,
    proportion_to_mutate: float,
    eta: float,
    minval: float,
    maxval: float,
) -> Tuple[Genotypes, RNGKey]:
    """
    Polynomial mutation over several genotypes

    Parameters:
        x (Genotypes): array of genotypes to transform (real values only)
        random_key (RNGKey): RNG key for reproducibility
        proportion_to_mutate (float): proportion of variables to mutate in each
            genotype (must be in [0, 1])
        eta (float): scaling parameter, the larger the more spread the new values will
            be
        minval (float): minimum value to clip the genotypes
        maxval (float): maximum value to clip the genotypes

    Returns:
        x (Genotypes): new genotypes
        random_key (RNGKey): new RNG key

    proportion to mutate between 0 and 1
    assumed to be of shape (batch_size, genotype_dim)
    random_key = random.split(random_key, num=x.shape[0])
    """
    mutation_key = jax.random.split(random_key, num=x.shape[0] * 2).reshape(
        x.shape[0], 2, 2
    )
    func = partial(
        _polynomial_mutation_function,
        proportion_to_mutate=proportion_to_mutate,
        eta=eta,
        minval=minval,
        maxval=maxval,
    )
    return jax.vmap(func)(x, mutation_key), mutation_key[0, 0]


def _polynomial_crossover_function(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    random_key: RNGKey,
    proportion_var_to_change: float,
):
    """
    Base crossover for one pair of genotypes

    x1 and x2 should have the same shape
    in this function we assume x1 shape and x2 shape to be (genotype_dim,)
    """
    num_var_to_change = int(proportion_var_to_change * x1.shape[0])
    indices = jnp.arange(start=0, stop=x1.shape[0])
    selected_indices = jax.random.choice(
        random_key, indices, shape=(num_var_to_change,)
    )
    x = x1.at[selected_indices].set(x2[selected_indices])
    return x


def polynomial_crossover_function(
    x1: Genotypes,
    x2: Genotypes,
    random_key: RNGKey,
    proportion_var_to_change: float,
) -> Tuple[Genotypes, RNGKey]:
    """
    Crossover over a set of pairs of genotypes

    Parameters:
        x1 (Genotypes): first batch of genotypes
        x2 (Genotypes): second batch of genotypes
        random_key (RNGKey): RNG key for reproducibility
        proportion_var_to_change (float): proportion of variables to exchange between
            genotypes (must be [0, 1])

    Returns:
        x (Genotypes): new genotypes
        random_key (RNGKey): new RNG key

    batched version of _simple_crossover_function
    x1 and x2 should have the same shape
    in this function we assume x1 shape and x2 shape to be (batch_size, genotype_dim)
    """
    crossover_key = jax.random.split(random_key, num=x2.shape[0])
    func = partial(
        _polynomial_crossover_function,
        proportion_var_to_change=proportion_var_to_change,
    )
    x = jax.vmap(func)(x1, x2, crossover_key)
    return x, crossover_key[0]


def isoline_crossover_function(
    x1: Genotypes,
    x2: Genotypes,
    random_key: RNGKey,
    iso_sigma: float,
    line_sigma: float,
    minval: Optional[float] = None,
    maxval: Optional[float] = None,
) -> Tuple[Genotypes, RNGKey]:
    """
    Iso+Line-DD Crossover Operator [1] over a set of pairs of genotypes

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

    [1] Vassiliades, Vassiiis, and Jean-Baptiste Mouret. "Discovering the elite
    hypervolume by leveraging interspecies correlation." Proceedings of the Genetic and
    Evolutionary Computation Conference. 2018.
    """

    key, subkey1, subkey2 = jax.random.split(random_key, num=3)
    iso_noise = jax.random.normal(subkey1, shape=x1.shape) * iso_sigma
    line_noise = jax.random.normal(subkey2, shape=x2.shape) * line_sigma
    x = (x1 + iso_noise) + line_noise * (x2 - x1)

    # Back in bounds if necessary (floating point issues)
    if (minval is not None) or (maxval is not None):
        x = jnp.clip(x, minval, maxval)

    return x, key
