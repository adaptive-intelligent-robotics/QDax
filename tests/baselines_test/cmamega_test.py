"""Tests CMA MEGA implementation"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import pytest

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from qdax.core.emitters.cma_mega_emitter import CMAMEGAEmitter
from qdax.core.map_elites import MAPElites
from qdax.custom_types import Descriptor, ExtraScores, Fitness, RNGKey


def test_cma_mega() -> None:

    num_iterations = 20000
    num_dimensions = 100
    num_centroids = 10000
    minval = -5.12
    maxval = 5.12
    batch_size = 36
    learning_rate = 1
    sigma_g = 10
    minval = -5.12
    maxval = 5.12

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

    rastrigin_grad_scores = jax.grad(rastrigin_scoring)

    def scoring_function(x: jnp.ndarray) -> Tuple[Fitness, Descriptor, ExtraScores]:
        scores, descriptors = rastrigin_scoring(x), rastrigin_descriptors(x)
        gradients = jnp.array(
            [
                rastrigin_grad_scores(x),
                jax.grad(_rastrigin_descriptor_1)(x),
                jax.grad(_rastrigin_descriptor_2)(x),
            ]
        ).T
        gradients = jnp.nan_to_num(gradients)

        # Compute normalized gradients
        norm_gradients = jax.tree_util.tree_map(
            lambda x: jnp.linalg.norm(x, axis=1, keepdims=True),
            gradients,
        )
        grads = jax.tree_util.tree_map(lambda x, y: x / y, gradients, norm_gradients)
        grads = jnp.nan_to_num(grads)
        extra_scores = {"gradients": gradients, "normalized_grads": grads}

        return scores, descriptors, extra_scores

    def scoring_fn(
        x: jnp.ndarray, random_key: RNGKey
    ) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
        fitnesses, descriptors, extra_scores = jax.vmap(scoring_function)(x)
        return fitnesses, descriptors, extra_scores, random_key

    worst_objective = rastrigin_scoring(-jnp.ones(num_dimensions) * 5.12)
    best_objective = rastrigin_scoring(jnp.ones(num_dimensions) * 5.12 * 0.4)

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

    random_key = jax.random.PRNGKey(0)
    initial_population = jax.random.uniform(
        random_key, shape=(batch_size, num_dimensions)
    )

    centroids, random_key = compute_cvt_centroids(
        num_descriptors=2,
        num_init_cvt_samples=10000,
        num_centroids=num_centroids,
        minval=minval,
        maxval=maxval,
        random_key=random_key,
    )

    emitter = CMAMEGAEmitter(
        scoring_function=scoring_fn,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_descriptors=2,
        centroids=centroids,
        sigma_g=sigma_g,
    )

    map_elites = MAPElites(
        scoring_function=scoring_fn, emitter=emitter, metrics_function=metrics_fn
    )
    repertoire, emitter_state, random_key = map_elites.init(
        initial_population, centroids, random_key
    )

    (
        repertoire,
        emitter_state,
        random_key,
    ), metrics = jax.lax.scan(
        map_elites.scan_update,
        (repertoire, emitter_state, random_key),
        (),
        length=num_iterations,
    )

    pytest.assume(repertoire is not None)


if __name__ == "__main__":
    test_cma_mega()
