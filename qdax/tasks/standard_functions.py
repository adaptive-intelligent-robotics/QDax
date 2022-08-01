from typing import Tuple

import jax
import jax.numpy as jnp

from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


def rastrigin(params: Genotype) -> Tuple[Fitness, Descriptor]:
    """
    2-D BD
    """
    x = params * 10 - 5  # scaling to [-5, 5]
    f = 10 * x.shape[0] + (x * x - 10 * jnp.cos(2 * jnp.pi * x)).sum()
    return -f, jnp.array([params[0], params[1]])


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
    Evaluate params in parallel
    """
    fitnesses, descriptors = jax.vmap(rastrigin)(params)

    return (fitnesses, descriptors, {}, random_key)


def sphere_scoring_function(
    params: Genotype,
    random_key: RNGKey,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Evaluate params in parallel
    """
    fitnesses, descriptors = jax.vmap(sphere)(params)

    return (fitnesses, descriptors, {}, random_key)


def rastrigin_proj_scoring(
    x: Genotype, minval: float, maxval: float
) -> Tuple[Fitness, Descriptor, ExtraScores]:
    def rastrigin_scoring(x: jnp.ndarray) -> jnp.ndarray:
        return -(
            10 * x.shape[-1]
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

    fitnesses, descriptors = rastrigin_scoring(x), rastrigin_descriptors(x)
    gradients = jnp.array(
        [
            rastrigin_grad_scores(x),
            jax.grad(_rastrigin_descriptor_1)(x),
            jax.grad(_rastrigin_descriptor_2)(x),
        ]
    ).T
    gradients = jnp.nan_to_num(gradients)

    return fitnesses, descriptors, {"gradients": gradients}


def rastrigin_proj_scoring_function(
    x: Genotype, random_key: RNGKey, minval: float = -5.12, maxval: float = 5.12
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:

    # vmap only over the Genotypes
    fitnesses, descriptors, extra_scores = jax.vmap(
        rastrigin_proj_scoring, in_axes=(0, None, None)
    )(x, minval, maxval)

    return fitnesses, descriptors, extra_scores, random_key
