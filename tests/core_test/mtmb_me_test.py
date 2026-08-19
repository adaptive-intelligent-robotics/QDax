"""Tests Multi-Task Multi-Behavior MAP-Elites implementation"""

import functools

import jax
import jax.numpy as jnp
import pytest

from qdax.core.containers.mapelites_repertoire import (
    compute_cvt_centroids,
    compute_euclidean_centroids,
    get_cells_indices,
)
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.mtmb_me import MTMBMAPElites
from qdax.tasks.arm import multi_task_multi_behavior_arm_scoring_function
from qdax.utils.metrics import default_mtmb_metrics


def test_mtmb_me() -> None:
    seed = 42
    num_param_dimensions = 10
    num_tasks = 16
    grid_shape = (8, 8)
    batch_size = 32
    num_iterations = 10

    key = jax.random.key(seed)

    # spread the tasks in the (link_length, max_angle) space with a CVT
    key, subkey = jax.random.split(key)
    task_descriptors = compute_cvt_centroids(
        num_descriptors=2,
        num_init_cvt_samples=1000,
        num_centroids=num_tasks,
        minval=0.1,
        maxval=1.0,
        key=subkey,
    )

    # behavior space tessellation, shared by all the tasks
    centroids = compute_euclidean_centroids(
        grid_shape=grid_shape,
        minval=0.0,
        maxval=1.0,
    )

    # Define emitter
    variation_fn = functools.partial(isoline_variation, iso_sigma=0.05, line_sigma=0.1)
    emitter = MixingEmitter(
        mutation_fn=lambda x, y: x,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )

    # Define a metrics function
    metrics_fn = functools.partial(
        default_mtmb_metrics, qd_offset=1.0, fitness_threshold=-0.1
    )

    # Instantiate MTMB MAP-Elites
    mtmb_me = MTMBMAPElites(
        scoring_function=multi_task_multi_behavior_arm_scoring_function,
        emitter=emitter,
        metrics_function=metrics_fn,
    )

    # Compute initial repertoire
    key, subkey = jax.random.split(key)
    init_genotypes = jax.random.uniform(
        subkey, shape=(batch_size, num_param_dimensions)
    )
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = mtmb_me.init(
        init_genotypes, centroids, task_descriptors, subkey
    )

    num_centroids = centroids.shape[0]
    pytest.assume(repertoire.fitnesses.shape == (num_tasks * num_centroids, 1))

    # Run the algorithm
    (
        repertoire,
        emitter_state,
        key,
    ), metrics = jax.lax.scan(
        mtmb_me.scan_update,
        (repertoire, emitter_state, key),
        (),
        length=num_iterations,
    )

    pytest.assume(repertoire is not None)

    # some tasks have elites
    pytest.assume(float(metrics["task_coverage"][-1]) > 0.0)
    pytest.assume(float(metrics["coverage"][-1]) > 0.0)

    # every occupied cell stores an elite whose behavior descriptor belongs
    # to the behavior cell of the row it is stored at
    occupied = repertoire.fitnesses[:, 0] != -jnp.inf
    behavior_cells = get_cells_indices(repertoire.descriptors, repertoire.centroids)
    expected_cells = jnp.arange(num_tasks * num_centroids) % num_centroids
    pytest.assume(bool(jnp.all(~occupied | (behavior_cells == expected_cells))))

    # the single-task view is a valid MAP-Elites repertoire
    task_repertoire = repertoire.get_task_repertoire(0)
    pytest.assume(task_repertoire.fitnesses.shape == (num_centroids, 1))
    pytest.assume(
        bool(jnp.all(task_repertoire.fitnesses == repertoire.fitnesses[:num_centroids]))
    )


def test_mtmb_me_ask_tell() -> None:
    seed = 42
    num_param_dimensions = 10
    num_tasks = 16
    grid_shape = (8, 8)
    batch_size = 32
    num_iterations = 5

    key = jax.random.key(seed)

    key, subkey = jax.random.split(key)
    task_descriptors = compute_cvt_centroids(
        num_descriptors=2,
        num_init_cvt_samples=1000,
        num_centroids=num_tasks,
        minval=0.1,
        maxval=1.0,
        key=subkey,
    )

    centroids = compute_euclidean_centroids(
        grid_shape=grid_shape,
        minval=0.0,
        maxval=1.0,
    )

    variation_fn = functools.partial(isoline_variation, iso_sigma=0.05, line_sigma=0.1)
    emitter = MixingEmitter(
        mutation_fn=lambda x, y: x,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )

    metrics_fn = functools.partial(default_mtmb_metrics, qd_offset=1.0)

    mtmb_me = MTMBMAPElites(
        scoring_function=None,
        emitter=emitter,
        metrics_function=metrics_fn,
    )

    # Evaluate the initial population
    key, subkey = jax.random.split(key)
    init_genotypes = jax.random.uniform(
        subkey, shape=(batch_size, num_param_dimensions)
    )
    key, subkey = jax.random.split(key)
    task_indices = jax.random.randint(subkey, (batch_size,), 0, num_tasks)
    key, subkey = jax.random.split(key)
    fitnesses, descriptors, extra_scores = (
        multi_task_multi_behavior_arm_scoring_function(
            init_genotypes, task_descriptors[task_indices], subkey
        )
    )

    repertoire, emitter_state, init_metrics = mtmb_me.init_ask_tell(
        genotypes=init_genotypes,
        fitnesses=fitnesses,
        descriptors=descriptors,
        task_indices=task_indices,
        centroids=centroids,
        task_descriptors=task_descriptors,
        key=key,
        extra_scores=extra_scores,
    )

    ask_fn = jax.jit(mtmb_me.ask)
    tell_fn = jax.jit(mtmb_me.tell)

    for _ in range(num_iterations):
        key, subkey = jax.random.split(key)
        genotypes, task_indices, extra_info = ask_fn(repertoire, emitter_state, subkey)

        key, subkey = jax.random.split(key)
        fitnesses, descriptors, extra_scores = (
            multi_task_multi_behavior_arm_scoring_function(
                genotypes, task_descriptors[task_indices], subkey
            )
        )

        repertoire, emitter_state, current_metrics = tell_fn(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            task_indices=task_indices,
            repertoire=repertoire,
            emitter_state=emitter_state,
            extra_scores=extra_scores,
            extra_info=extra_info,
        )

    pytest.assume(repertoire is not None)


if __name__ == "__main__":
    test_mtmb_me()
    test_mtmb_me_ask_tell()
