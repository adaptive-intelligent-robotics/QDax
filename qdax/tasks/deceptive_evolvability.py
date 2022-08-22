from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from qdax.tasks.archimedean_spiral import QDBenchmarkTask
from qdax.types import Genotype, Fitness, Descriptor


def multivariate_normal(params: Genotype,
                        mu: jnp.ndarray,
                        sigma: float
                        ):
    """
    Compute the unnormalised multivariate normal density.
    """
    params = params.reshape((-1, 1))
    mu = mu.reshape((-1, 1))

    x = params - mu
    return jnp.exp(-0.5 * x.T.dot(x).ravel() / (sigma * sigma)) \
           * (2 * jnp.pi * sigma * sigma) ** (-params.shape[0] / 2)


def _get_coeff(sigma,
               dim=2
               ):
    return (2 * jnp.pi * sigma * sigma) ** (-dim / 2)


class DeceptiveEvolvabilityV0(QDBenchmarkTask):
    def __init__(self,
                 mu_1=jnp.array([50., 125.]),
                 sigma_1=jnp.sqrt(70.),
                 beta=20.,
                 mu_2=jnp.array([150., 125.]),
                 sigma_2=jnp.sqrt(1e3),
                 ):
        self.mu_1 = mu_1
        self.sigma_1 = sigma_1
        self.beta = beta
        self.mu_2 = mu_2
        self.sigma_2 = sigma_2

    def scoring_function(self,
                         params: Genotype
                         ) -> Tuple[Fitness, Descriptor]:
        bd = multivariate_normal(params,
                                 self.mu_1,
                                 self.sigma_1) \
             + self.beta * multivariate_normal(params,
                                               self.mu_2,
                                               self.sigma_2)
        constant_fitness = jnp.asarray(1.0)
        return constant_fitness, bd

    def get_saddle_point(self):
        func_to_minimize = lambda theta: self.scoring_function(theta)[1]

        t = jnp.linspace(0.,
                         1.,
                         1000)

        considered_points = jax.vmap(lambda _t: _t * (
              self.mu_2 - self.mu_1) + self.mu_1)(t)

        results = jax.vmap(func_to_minimize)(considered_points)

        index_min = jnp.argmin(results)

        return considered_points[index_min]

    def get_bd_size(self) -> int:
        return 1

    def get_min_max_bd(self):
        potential_max_1 = self.scoring_function(self.mu_1)[1]
        potential_max_2 = self.scoring_function(self.mu_2)[1]
        return 0., jnp.max([potential_max_1, potential_max_2])[0]

    def get_min_max_params(self):
        potential_max_1 = jnp.max(self.mu_1 + 3 * self.sigma_1)
        potential_max_2 = jnp.max(self.mu_2 + 3 * self.sigma_2)
        max_final = jnp.maximum(potential_max_1,
                                potential_max_2)

        potential_min_1 = jnp.min(self.mu_1 - 3 * self.sigma_1)
        potential_min_2 = jnp.min(self.mu_2 - 3 * self.sigma_2)
        min_final = jnp.minimum(potential_min_1,
                                potential_min_2)
        return min_final, max_final

    def get_initial_parameters(self,
                               batch_size: int
                               ) -> Genotype:
        saddle_point = self.get_saddle_point()
        return jnp.repeat(jnp.expand_dims(saddle_point,
                                          axis=0),
                          batch_size)


def function_test():
    task = DeceptiveEvolvabilityV0()
    saddle_point = task.get_saddle_point()
    min_final, max_final = task.get_min_max_params()
    x = jnp.linspace(min_final,
                     max_final,
                     200)

    # numpy.linspace creates an array of
    # 9 linearly placed elements between
    # -4 and 4, both inclusive
    y = jnp.linspace(min_final,
                     max_final,
                     200)

    # The meshgrid function returns
    # two 2-dimensional arrays
    x_1, y_1 = jnp.meshgrid(x,
                            y)

    all_points = jnp.stack((x_1, y_1),
                           axis=2)
    results = jax.vmap(jax.vmap(lambda theta: task.scoring_function(theta)[
        1]))(all_points)
    print(results.shape)
    plt.pcolor(x_1,
               y_1,
               results.squeeze(axis=2))
    print(results)
    plt.scatter(saddle_point[0],
                saddle_point[1],
                c='r')
    plt.show()

    # deceptive_evolvability_v0()

    print(_get_coeff(sigma=task.sigma_1))
    print(_get_coeff(sigma=task.sigma_2))


if __name__ == '__main__':
    function_test()
