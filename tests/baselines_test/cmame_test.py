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
from qdax.types import Descriptor, ExtraScores, Fitness, RNGKey


@pytest.mark.parametrize(
    "emitter_type",
    [CMAOptimizingEmitter, CMARndEmitter, CMAImprovementEmitter],
)
def test_cma_me(emitter_type: Type[CMAEmitter]) -> None:

    num_iterations = 1000
    num_dimensions = 20
    grid_shape = (50, 50)
    batch_size = 36
    sigma_g = 0.5
    minval = -5.12
    maxval = 5.12
    min_bd = -5.12 * 0.5 * num_dimensions
    max_bd = 5.12 * 0.5 * num_dimensions
    pool_size = 3

    def sphere_scoring(x: jnp.ndarray) -> jnp.ndarray:
        return -jnp.sum((x + minval * 0.4) * (x + minval * 0.4), axis=-1)

    fitness_scoring = sphere_scoring

    def clip(x: jnp.ndarray) -> jnp.ndarray:
        in_bound = (x <= maxval) * (x >= minval)
        return jnp.where(condition=in_bound, x=x, y=(maxval / x))

    def _behavior_descriptor_1(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(clip(x[: x.shape[-1] // 2]))

    def _behavior_descriptor_2(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(clip(x[x.shape[-1] // 2 :]))

    def _behavior_descriptors(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([_behavior_descriptor_1(x), _behavior_descriptor_2(x)])

    def scoring_function(x: jnp.ndarray) -> Tuple[Fitness, Descriptor, Dict]:
        scores, descriptors = fitness_scoring(x), _behavior_descriptors(x)
        return scores, descriptors, {}

    def scoring_fn(
        x: jnp.ndarray, random_key: RNGKey
    ) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
        fitnesses, descriptors, extra_scores = jax.vmap(scoring_function)(x)
        return fitnesses, descriptors, extra_scores, random_key

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

    random_key = jax.random.PRNGKey(0)
    initial_population = (
        jax.random.uniform(random_key, shape=(batch_size, num_dimensions)) * 0.0
    )

    centroids = compute_euclidean_centroids(
        grid_shape=grid_shape,
        minval=min_bd,
        maxval=max_bd,
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

    repertoire, emitter_state, random_key = map_elites.init(
        initial_population, centroids, random_key
    )

    (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
        map_elites.scan_update,
        (repertoire, emitter_state, random_key),
        (),
        length=num_iterations,
    )

    pytest.assume(metrics["coverage"][-1] > 25)
    pytest.assume(metrics["max_fitness"][-1] > 95)
    pytest.assume(metrics["qd_score"][-1] > 50000)


if __name__ == "__main__":
    test_cma_me(emitter_type=CMAEmitter)
