from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from qdax.tasks.archimedean_spiral import QDBenchmarkTask
from qdax.types import Descriptor, Fitness, Genotype


def multivariate_normal(
    params: Genotype,
    mu: jnp.ndarray,
    sigma: float,
) -> jnp.ndarray:
    """
    Compute the un-normalised multivariate normal density.
    """
    params = params.reshape((-1, 1))
    mu = mu.reshape((-1, 1))

    x = params - mu
    return jnp.exp(-0.5 * x.T.dot(x).ravel() / (sigma * sigma)) * (
        2 * jnp.pi * sigma * sigma
    ) ** (-params.shape[0] / 2)


class DeceptiveEvolvabilityV0(QDBenchmarkTask):
    default_mu_1 = jnp.array([50.0, 125.0])
    default_sigma_1 = jnp.sqrt(70.0)
    default_beta = 20.0
    default_mu_2 = jnp.array([150.0, 125.0])
    default_sigma_2 = jnp.sqrt(1e3)

    def __init__(
        self,
        mu_1: Optional[Genotype] = None,
        sigma_1: Optional[float] = None,
        beta: Optional[float] = None,
        mu_2: Optional[Genotype] = None,
        sigma_2: Optional[float] = None,
    ):
        if mu_1 is None:
            mu_1 = self.default_mu_1
        if sigma_1 is None:
            sigma_1 = self.default_sigma_1
        if beta is None:
            beta = self.default_beta
        if mu_2 is None:
            mu_2 = self.default_mu_2
        if sigma_2 is None:
            sigma_2 = self.default_sigma_2

        self.mu_1 = mu_1
        self.sigma_1 = sigma_1
        self.beta = beta
        self.mu_2 = mu_2
        self.sigma_2 = sigma_2

    def evaluation(self, params: Genotype) -> Tuple[Fitness, Descriptor]:
        bd = multivariate_normal(
            params, self.mu_1, self.sigma_1
        ) + self.beta * multivariate_normal(params, self.mu_2, self.sigma_2)
        constant_fitness = jnp.asarray(1.0)
        return constant_fitness, bd

    def get_saddle_point(self) -> Genotype:
        def _func_to_minimize(theta: Genotype) -> Descriptor:
            return self.evaluation(theta)[1]

        t = jnp.linspace(0.0, 1.0, 1000)

        considered_points = jax.vmap(
            lambda _t: _t * (self.mu_2 - self.mu_1) + self.mu_1
        )(t)

        results = jax.vmap(_func_to_minimize)(considered_points)

        index_min = jnp.argmin(results)

        return considered_points[index_min]

    def get_bd_size(self) -> int:
        return 1

    def get_min_max_bd(self) -> Tuple[float, float]:
        potential_max_1 = self.evaluation(self.mu_1)[1]
        potential_max_2 = self.evaluation(self.mu_2)[1]
        return 0.0, jnp.maximum(potential_max_1, potential_max_2)[0]

    def get_min_max_params(self) -> Tuple[float, float]:
        potential_max_1 = jnp.max(self.mu_1 + 3 * self.sigma_1)
        potential_max_2 = jnp.max(self.mu_2 + 3 * self.sigma_2)
        max_final = jnp.maximum(potential_max_1, potential_max_2)

        potential_min_1 = jnp.min(self.mu_1 - 3 * self.sigma_1)
        potential_min_2 = jnp.min(self.mu_2 - 3 * self.sigma_2)
        min_final = jnp.minimum(potential_min_1, potential_min_2)
        return min_final, max_final

    def get_initial_parameters(self, batch_size: int) -> Genotype:
        saddle_point = self.get_saddle_point()
        return jnp.repeat(jnp.expand_dims(saddle_point, axis=0), batch_size, axis=0)


deceptive_evolvability_v0_task = DeceptiveEvolvabilityV0()


def function_test() -> None:
    task = DeceptiveEvolvabilityV0()
    saddle_point = task.get_saddle_point()
    min_final, max_final = task.get_min_max_params()
    x = jnp.linspace(min_final, max_final, 200)

    # numpy.linspace creates an array of
    # 9 linearly placed elements between
    # -4 and 4, both inclusive
    y = jnp.linspace(min_final, max_final, 200)

    # The meshgrid function returns
    # two 2-dimensional arrays
    x_1, y_1 = jnp.meshgrid(x, y)

    all_points = jnp.stack((x_1, y_1), axis=2)
    results = jax.vmap(jax.vmap(lambda theta: task.evaluation(theta)[1]))(all_points)
    print(results.shape)
    plt.pcolor(x_1, y_1, results.squeeze(axis=2))
    print(results)
    plt.scatter(saddle_point[0], saddle_point[1], c="r")
    plt.show()


if __name__ == "__main__":
    function_test()
