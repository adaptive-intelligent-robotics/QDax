import math
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import pytest

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_euclidean_centroids,
)
from qdax.core.emitters.omg_mega_emitter import OMGMEGAEmitter
from qdax.core.map_elites import MAPElites
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


def test_omg_mega() -> None:

    num_iterations = 200
    num_dimensions = 1000
    num_centroids = 10000
    num_descriptors = 2
    sigma_g = 10
    minval = -5.12
    maxval = 5.12
    batch_size = 36

    def rastrigin_scoring(x: jnp.ndarray) -> Fitness:
        return -(
            10 * x.shape[-1]
            + jnp.sum(
                (x + minval * 0.4) ** 2 - 10 * jnp.cos(2 * jnp.pi * (x + minval * 0.4))
            )
        )

    def clip(x: jnp.ndarray) -> jnp.ndarray:
        return x * (x <= maxval) * (x >= minval) + maxval / x * (
            (x > maxval) + (x < minval)
        )

    def _rastrigin_descriptor_1(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(clip(x[: x.shape[0] // 2]))

    def _rastrigin_descriptor_2(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(clip(x[x.shape[0] // 2 :]))

    def rastrigin_descriptors(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([_rastrigin_descriptor_1(x), _rastrigin_descriptor_2(x)])

    rastrigin_grad_scores = jax.grad(rastrigin_scoring)

    def scoring_function(x: jnp.ndarray) -> Tuple[Fitness, Descriptor, ExtraScores]:
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

    def scoring_fn(x: Genotype, key: RNGKey) -> Tuple[Fitness, Descriptor, ExtraScores]:
        fitnesses, descriptors, extra_scores = jax.vmap(scoring_function)(x)
        return fitnesses, descriptors, extra_scores

    worst_objective = rastrigin_scoring(-jnp.ones(num_dimensions) * maxval)
    best_objective = rastrigin_scoring(jnp.ones(num_dimensions) * maxval * 0.4)

    def metrics_fn(repertoire: MapElitesRepertoire) -> Dict[str, jnp.ndarray]:

        # get metrics
        grid_empty = repertoire.fitnesses == -jnp.inf
        adjusted_fitness = (repertoire.fitnesses - worst_objective) / (
            best_objective - worst_objective
        )
        qd_score = jnp.sum(adjusted_fitness, where=~grid_empty) / num_centroids
        coverage = 100 * jnp.mean(1.0 - grid_empty)
        max_fitness = jnp.max(adjusted_fitness)
        return {"qd_score": qd_score, "max_fitness": max_fitness, "coverage": coverage}

    key = jax.random.key(0)

    # defines the population
    key, subkey = jax.random.split(key)
    initial_population = jax.random.uniform(subkey, shape=(100, num_dimensions))

    sqrt_centroids = int(math.sqrt(num_centroids))  # 2-D grid
    grid_shape = (sqrt_centroids, sqrt_centroids)
    centroids = compute_euclidean_centroids(
        grid_shape=grid_shape,
        minval=minval,
        maxval=maxval,
    )

    # defines the emitter
    emitter = OMGMEGAEmitter(
        batch_size=batch_size,
        sigma_g=sigma_g,
        num_descriptors=num_descriptors,
        centroids=centroids,
    )

    # create the MAP Elites instance
    map_elites = MAPElites(
        scoring_function=scoring_fn, emitter=emitter, metrics_function=metrics_fn
    )

    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = map_elites.init(
        initial_population, centroids, subkey
    )

    (
        repertoire,
        emitter_state,
        key,
    ), metrics = jax.lax.scan(
        map_elites.scan_update,
        (repertoire, emitter_state, key),
        (),
        length=num_iterations,
    )

    pytest.assume(metrics["coverage"][-1] > 5)
    pytest.assume(metrics["max_fitness"][-1] > 0.5)
    pytest.assume(metrics["qd_score"][-1] > 0.05)


if __name__ == "__main__":
    test_omg_mega()
