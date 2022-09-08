from typing import Tuple

import jax
import jax.numpy as jnp

from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


def rastrigin(params: Genotype) -> Tuple[Fitness, Descriptor]:
    """
    2-D BD
    """
    x = params * 10 - 5  # scaling to [-5, 5]
    f = jnp.asarray(10.0 * x.shape[0]) + jnp.sum(x * x - 10 * jnp.cos(2 * jnp.pi * x))
    return -f, jnp.asarray([params[0], params[1]])


def sphere(params: Genotype) -> Tuple[Fitness, Descriptor]:
    """
    2-D BD
    """
    x = params * 10 - 5  # scaling to [-5, 5]
    f = (x * x).sum()
    return -f, jnp.array([params[0], params[1]])


def rastrigin_scoring_function(
    params: Genotype,
    random_key: RNGKey,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Scoring function for the rastrigin function
    """
    fitnesses, descriptors = jax.vmap(rastrigin)(params)

    return fitnesses, descriptors, {}, random_key


def sphere_scoring_function(
    params: Genotype,
    random_key: RNGKey,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Scoring function for the sphere function
    """
    fitnesses, descriptors = jax.vmap(sphere)(params)

    return fitnesses, descriptors, {}, random_key


def _rastrigin_proj_scoring(
    params: Genotype, minval: float, maxval: float
) -> Tuple[Fitness, Descriptor, ExtraScores]:
    """
    Rastrigin function with a folding of the behaviour space.

    Args:
        params: Genotype
        minval: minimum value of the parameters
        maxval: maximum value of the parameters

    Returns:
        fitnesses
        descriptors
        extra_scores (containing the gradients of the
            fitnesses and descriptors)
    """

    def rastrigin_scoring(x: jnp.ndarray) -> jnp.ndarray:
        return -(
            jnp.asarray(10 * x.shape[-1])
            + jnp.sum(
                (x + minval * 0.4) ** 2 - 10 * jnp.cos(2 * jnp.pi * (x + minval * 0.4))
            )
        )

    def clip(x: jnp.ndarray) -> jnp.ndarray:
        return x * (x <= maxval) * (x >= +minval) + maxval / x * (
            (x > maxval) + (x < +minval)
        )

    def _rastrigin_descriptor_1(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(clip(x[: x.shape[0] // 2]))

    def _rastrigin_descriptor_2(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(clip(x[x.shape[0] // 2 :]))

    def rastrigin_descriptors(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([_rastrigin_descriptor_1(x), _rastrigin_descriptor_2(x)])

    # gradient function
    rastrigin_grad_scores = jax.grad(rastrigin_scoring)

    fitnesses, descriptors = rastrigin_scoring(params), rastrigin_descriptors(params)
    gradients = jnp.array(
        [
            rastrigin_grad_scores(params),
            jax.grad(_rastrigin_descriptor_1)(params),
            jax.grad(_rastrigin_descriptor_2)(params),
        ]
    ).T
    gradients = jnp.nan_to_num(gradients)

    return fitnesses, descriptors, {"gradients": gradients}


def rastrigin_proj_scoring_function(
    params: Genotype, random_key: RNGKey, minval: float = -5.12, maxval: float = 5.12
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Scoring function for the rastrigin function with
    a folding of the behaviour space.
    """

    # vmap only over the Genotypes
    fitnesses, descriptors, extra_scores = jax.vmap(
        _rastrigin_proj_scoring, in_axes=(0, None, None)
    )(params, minval, maxval)

    return fitnesses, descriptors, extra_scores, random_key
