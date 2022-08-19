from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


def square(params: Genotype) -> Tuple[Fitness, Descriptor]:
    """
    Seach space should be [0,1]^n
    BD space should be [0,1]^n
    GETTING THE EXPECTED RESULT BUT DIFFERENT FROM PAPER FUNCTION
    - PAPER FUNCTION MIGHT BE WRONG
    jnp.sin(coeff*params) --> params
    """
    # coeff = 5
    f = 1 - jnp.prod(params)
    # bd = jnp.sin(coeff*params) # from the paper but not working
    bd = params
    return f, bd


def checkered(params: Genotype) -> Tuple[Fitness, Descriptor]:
    """
    Seach space should be [0,1]^n
    BD space should be [0,1]^n
    GETTING THE EXPECTED RESULT BUT DIFFERENT FROM PAPER FUNCTION
    - PAPER FUNCTION MIGHT BE WRONG
    """
    coeff = 10
    f = jnp.prod(jnp.sin(coeff * params * jnp.pi))
    bd = jnp.sin(params * jnp.pi)
    return f, bd


def empty_circle(params: Genotype) -> Tuple[Fitness, Descriptor]:
    """
    Seach space should be [0,1]^n
    BD space should be [0,1]^n
    TODO: NOT WORKING - PLOT DOESNT GIVE EXPECTED RESULT
    """
    coeff = 40
    f = jnp.exp(-((jnp.linalg.norm(params - jnp.ones_like(params) * 0.5) - 0.5) ** 2))
    bd = jnp.sin(coeff * params * jnp.pi)
    return f, bd


def non_continous_islands(params: Genotype) -> Tuple[Fitness, Descriptor]:
    """
    Seach space should be [0,1]^n
    BD space should be [0,1]^n
    """
    f = jnp.prod(params)
    bd = jnp.round(10 * params) / 10
    return f, bd


def continous_islands(params: Genotype) -> Tuple[Fitness, Descriptor]:
    """
    Seach space should be [0,1]^n
    BD space should be [0,1]^n
    GETTING THE EXPECTED RESULT BUT DIFFERENT FROM PAPER FUNCTION
    - PAPER FUNCTION MIGHT BE WRONG
    remove pi from denominator of BD inside the sin
    """
    coeff = 20
    f = jnp.prod(params)
    bd = params - jnp.sin((coeff * jnp.pi * params) / 20)
    return f, bd


def get_scoring_function(
    task_fn: Callable[[Genotype], Tuple[Fitness, Descriptor]]
) -> Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]]:
    def scoring_function(
        params: Genotype,
        random_key: RNGKey,
    ) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
        """
        Evaluate params in parallel
        """
        fitnesses, descriptors = jax.vmap(task_fn)(params)

        return (fitnesses, descriptors, {}, random_key)

    return scoring_function


square_scoring_function = get_scoring_function(square)
checkered_scoring_function = get_scoring_function(checkered)
empty_circle_scoring_function = get_scoring_function(empty_circle)
non_continous_islands_scoring_function = get_scoring_function(non_continous_islands)
continous_islands_scoring_function = get_scoring_function(continous_islands)
