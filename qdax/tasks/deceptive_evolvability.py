from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from qdax.types import Genotype, Fitness, Descriptor


def unnormalised_multivariate_normal(params: Genotype,
                                     mu: jnp.ndarray,
                                     sigma: float):
    """
    Compute the unnormalised multivariate normal density.
    """
    params = params.reshape((-1, 1))
    mu = mu.reshape((-1, 1))

    x = params - mu
    return jnp.exp(-0.5 * x.T.dot(x).ravel() / (sigma * sigma))

def deceptive_evolvability_v0(params: Genotype,
                              mu_1,
                              sigma_1,
                              beta,
                              mu_2,
                              sigma_2,) -> Tuple[Fitness, Descriptor]:
    bd = unnormalised_multivariate_normal(params, mu_1, sigma_1) + beta * unnormalised_multivariate_normal(params, mu_2, sigma_2)
    constant_fitness = jnp.asarray(1.0)
    print(bd)
    return constant_fitness, bd

def function_test():
    mu_1 = jnp.array([50., 125.])
    mu_2 = jnp.array([150., 125.])
    sigma_1 = jnp.sqrt(70.)
    sigma_2 = jnp.sqrt(1e3)
    beta = 2
    x = jnp.linspace(0.,
                     180,
                     100)

    # numpy.linspace creates an array of
    # 9 linearly placed elements between
    # -4 and 4, both inclusive
    y = jnp.linspace(0.,
                     180,
                     100)

    # The meshgrid function returns
    # two 2-dimensional arrays
    x_1, y_1 = jnp.meshgrid(x,
                           y)

    f = jax.vmap(jax.vmap(partial(deceptive_evolvability_v0, mu_1=mu_1, sigma_1=sigma_1, beta=beta, mu_2=mu_2, sigma_2=sigma_2)))

    all_points = jnp.stack((x_1, y_1), axis=2)
    print(all_points.shape)
    fitness, bd = f(all_points)

    # Build the plot
    print(bd.shape)
    plt.pcolor(x_1, y_1, bd.squeeze(axis=2))
    plt.show()

    # deceptive_evolvability_v0()


if __name__ == '__main__':
    function_test()
