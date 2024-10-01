"""Tests CMA ME implementation"""

from typing import Dict, Tuple, Type

import jax
import jax.numpy as jnp
import pytest

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_euclidean_centroids,
)
from qdax.core.emitters.cma_emitter import CMAEmitter
from qdax.core.emitters.cma_improvement_emitter import CMAImprovementEmitter
from qdax.core.emitters.cma_opt_emitter import CMAOptimizingEmitter
from qdax.core.emitters.cma_pool_emitter import CMAPoolEmitter
from qdax.core.emitters.cma_rnd_emitter import CMARndEmitter
from qdax.core.map_elites import MAPElites
from qdax.custom_types import Descriptor, ExtraScores, Fitness, RNGKey


@pytest.mark.parametrize(
    "emitter_type",
    [CMAOptimizingEmitter, CMARndEmitter, CMAImprovementEmitter],
)
def test_cma_me(emitter_type: Type[CMAEmitter]) -> None:

    num_iterations = 2000
    num_dimensions = 20
    grid_shape = (50, 50)
    batch_size = 36
    sigma_g = 0.5
    minval = -5.12
    maxval = 5.12
    min_descriptor = -5.12 * 0.5 * num_dimensions
    max_descriptor = 5.12 * 0.5 * num_dimensions
    pool_size = 3

    def sphere_scoring(x: jnp.ndarray) -> jnp.ndarray:
        return -jnp.sum((x + minval * 0.4) * (x + minval * 0.4), axis=-1)

    fitness_scoring = sphere_scoring

    def clip(x: jnp.ndarray) -> jnp.ndarray:
        in_bound = (x <= maxval) * (x >= minval)
        return jnp.where(in_bound, x, (maxval / x))

    def _descriptor_1(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(clip(x[: x.shape[-1] // 2]))

    def _descriptor_2(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(clip(x[x.shape[-1] // 2 :]))

    def _descriptors(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([_descriptor_1(x), _descriptor_2(x)])

    def scoring_function(x: jnp.ndarray) -> Tuple[Fitness, Descriptor, Dict]:
        scores, descriptors = fitness_scoring(x), _descriptors(x)
        return scores, descriptors, {}

    def scoring_fn(
        x: jnp.ndarray, key: RNGKey
    ) -> Tuple[Fitness, Descriptor, ExtraScores]:
        fitnesses, descriptors, extra_scores = jax.vmap(scoring_function)(x)
        return fitnesses, descriptors, extra_scores

    worst_objective = fitness_scoring(-jnp.ones(num_dimensions) * 5.12)
    best_objective = fitness_scoring(jnp.ones(num_dimensions) * 5.12 * 0.4)

    def metrics_fn(repertoire: MapElitesRepertoire) -> Dict[str, jnp.ndarray]:

        # get metrics
        grid_empty = repertoire.fitnesses == -jnp.inf
        adjusted_fitness = (
            (repertoire.fitnesses - worst_objective)
            * 100
            / (best_objective - worst_objective)
        )
        qd_score = jnp.sum(adjusted_fitness, where=~grid_empty)  # / num_centroids
        coverage = 100 * jnp.mean(1.0 - grid_empty)
        max_fitness = jnp.max(adjusted_fitness)
        return {"qd_score": qd_score, "max_fitness": max_fitness, "coverage": coverage}

    key = jax.random.key(0)

    key, subkey = jax.random.split(key)
    initial_population = (
        jax.random.uniform(subkey, shape=(batch_size, num_dimensions)) * 0.0
    )

    centroids = compute_euclidean_centroids(
        grid_shape=grid_shape,
        minval=min_descriptor,
        maxval=max_descriptor,
    )

    emitter_kwargs = {
        "batch_size": batch_size,
        "genotype_dim": num_dimensions,
        "centroids": centroids,
        "sigma_g": sigma_g,
        "min_count": 1,
        "max_count": None,
    }

    emitter = emitter_type(**emitter_kwargs)

    emitter = CMAPoolEmitter(num_states=pool_size, emitter=emitter)

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

    pytest.assume(metrics["coverage"][-1] > 25)
    pytest.assume(metrics["max_fitness"][-1] > 95)
    pytest.assume(metrics["qd_score"][-1] > 50000)


if __name__ == "__main__":
    test_cma_me(emitter_type=CMAEmitter)
