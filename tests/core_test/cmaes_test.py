"""Tests CMA ES implementation"""

import jax
import jax.numpy as jnp
import pytest

from qdax.core.cmaes import CMAES


def test_cmaes() -> None:

    num_iterations = 10000
    num_dimensions = 100
    batch_size = 36
    num_best = 36
    sigma_g = 0.5
    minval = -5.12

    def sphere_scoring(x: jnp.ndarray) -> jnp.ndarray:
        return -jnp.sum((x + minval * 0.4) * (x + minval * 0.4), axis=-1)

    fitness_fn = sphere_scoring

    cmaes = CMAES(
        population_size=batch_size,
        num_best=num_best,
        search_dim=num_dimensions,
        fitness_function=fitness_fn,  # type: ignore
        mean_init=jnp.zeros((num_dimensions,)),
        init_sigma=sigma_g,
        delay_eigen_decomposition=True,
    )

    state = cmaes.init()
    random_key = jax.random.PRNGKey(0)

    iteration_count = 0
    for _ in range(num_iterations):
        iteration_count += 1

        # sample
        samples, random_key = cmaes.sample(state, random_key)

        # udpate
        state = cmaes.update(state, samples)

        # check stop condition
        stop_condition = cmaes.stop_condition(state)

        if stop_condition:
            break

    fitnesses = fitness_fn(samples)

    pytest.assume(jnp.min(fitnesses) < 0.001)


if __name__ == "__main__":
    test_cmaes()
