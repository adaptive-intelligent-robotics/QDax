"""
Hypervolume Benchmark Functions in the paper by
J.B. Mouret, "Hypervolume-based Benchmark Functions for Quality Diversity Algorithms"
"""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


def square(params: Genotype) -> Tuple[Fitness, Descriptor]:
    """
    Search space should be [0,1]^n
    Descriptor space should be [0,1]^n
    """
    freq = 5
    f = 1 - jnp.prod(params)
    descriptor = jnp.sin(freq * params)
    return f, descriptor


def checkered(params: Genotype) -> Tuple[Fitness, Descriptor]:
    """
    Search space should be [0,1]^n
    Descriptor space should be [0,1]^n
    """
    freq = 5
    f = jnp.prod(jnp.sin(params * 50))
    descriptor = jnp.sin(params * freq)
    return f, descriptor


def empty_circle(params: Genotype) -> Tuple[Fitness, Descriptor]:
    """
    Search space should be [0,1]^n
    Descriptor space should be [0,1]^n
    """

    def _gaussian(x: jnp.ndarray, mu: float, sig: float) -> jnp.ndarray:
        return jnp.exp(-jnp.power(x - mu, 2.0) / (2 * jnp.power(sig, 2.0)))

    freq = 40
    centre = jnp.ones_like(params) * 0.5
    distance_from_centre = jnp.linalg.norm(params - centre)
    f = _gaussian(distance_from_centre, mu=0.5, sig=0.3)
    descriptor = jnp.sin(freq * params)
    return f, descriptor


def non_continous_islands(params: Genotype) -> Tuple[Fitness, Descriptor]:
    """
    Search space should be [0,1]^n
    Descriptor space should be [0,1]^n
    """
    f = jnp.prod(params)
    descriptor = jnp.round(10 * params) / 10
    return f, descriptor


def continous_islands(params: Genotype) -> Tuple[Fitness, Descriptor]:
    """
    Search space should be [0,1]^n
    Descriptor space should be [0,1]^n
    """
    coeff = 20
    f = jnp.prod(params)
    descriptor = params - jnp.sin(coeff * jnp.pi * params) / (coeff * jnp.pi)
    return f, descriptor


def get_scoring_function(
    task_fn: Callable[[Genotype], Tuple[Fitness, Descriptor]]
) -> Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores]]:

    def scoring_function(
        params: Genotype,
        key: RNGKey,
    ) -> Tuple[Fitness, Descriptor, ExtraScores]:
        """
        Evaluate params in parallel
        """
        fitnesses, descriptors = jax.vmap(task_fn)(params)

        return fitnesses, descriptors, {}

    return scoring_function


square_scoring_function = get_scoring_function(square)
checkered_scoring_function = get_scoring_function(checkered)
empty_circle_scoring_function = get_scoring_function(empty_circle)
non_continous_islands_scoring_function = get_scoring_function(non_continous_islands)
continous_islands_scoring_function = get_scoring_function(continous_islands)
