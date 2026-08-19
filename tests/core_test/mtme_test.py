"""Tests Multi-task MAP-Elites implementation"""

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import pytest

from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.mtme_emitter import MultiTaskMixingEmitter
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.mtme import MultiTaskMAPElites
from qdax.tasks.arm import multi_task_arm_scoring_function
from qdax.utils.metrics import default_qd_metrics


def get_task_descriptors(key: jax.Array, num_tasks: int) -> jax.Array:
    """Spread the tasks in the (link_length, max_angle) space with a CVT."""
    return compute_cvt_centroids(
        num_descriptors=2,
        num_init_cvt_samples=1000,
        num_centroids=num_tasks,
        minval=0.1,
        maxval=1.0,
        key=key,
    )


@pytest.mark.parametrize(
    "tournament_sizes",
    [(1,), (1, 5, 10, 50)],
)
def test_mtme(tournament_sizes: Tuple[int, ...]) -> None:
    seed = 42
    num_param_dimensions = 10
    num_tasks = 64
    batch_size = 32
    num_iterations = 10

    key = jax.random.key(seed)

    key, subkey = jax.random.split(key)
    task_descriptors = get_task_descriptors(subkey, num_tasks)

    # Define emitter
    variation_fn = functools.partial(isoline_variation, iso_sigma=0.05, line_sigma=0.1)
    emitter = MultiTaskMixingEmitter(
        mutation_fn=lambda x, y: x,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )

    # Define a metrics function
    metrics_fn = functools.partial(default_qd_metrics, qd_offset=2.0)

    # Instantiate Multi-task MAP-Elites
    mtme = MultiTaskMAPElites(
        scoring_function=multi_task_arm_scoring_function,
        emitter=emitter,
        metrics_function=metrics_fn,
        tournament_sizes=tournament_sizes,
    )

    # Compute initial repertoire
    key, subkey = jax.random.split(key)
    init_genotypes = jax.random.uniform(
        subkey, shape=(batch_size, num_param_dimensions)
    )
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, mtme_state, init_metrics = mtme.init(
        init_genotypes, task_descriptors, subkey
    )

    # Run the algorithm
    (
        repertoire,
        emitter_state,
        mtme_state,
        key,
    ), metrics = jax.lax.scan(
        mtme.scan_update,
        (repertoire, emitter_state, mtme_state, key),
        (),
        length=num_iterations,
    )

    pytest.assume(repertoire is not None)

    # the bandit has been updated once per generation
    pytest.assume(int(jnp.sum(mtme_state.selection_counts)) == num_iterations)
    pytest.assume(int(mtme_state.generation) == num_iterations)

    # some cells have been filled
    pytest.assume(float(metrics["coverage"][-1]) > 0.0)

    # every occupied cell stores exactly the descriptor of its task
    occupied = repertoire.fitnesses[:, 0] != -jnp.inf
    descriptors_match = jnp.all(
        jnp.where(
            occupied[:, None],
            repertoire.descriptors == repertoire.centroids,
            True,
        )
    )
    pytest.assume(bool(descriptors_match))


def test_mtme_uniform_task_selection_with_standard_emitter() -> None:
    """With tournament_sizes=(1,), any emitter can be used."""
    seed = 0
    num_param_dimensions = 8
    num_tasks = 32
    batch_size = 16
    num_iterations = 5

    key = jax.random.key(seed)

    key, subkey = jax.random.split(key)
    task_descriptors = get_task_descriptors(subkey, num_tasks)

    variation_fn = functools.partial(isoline_variation, iso_sigma=0.05, line_sigma=0.1)
    emitter = MixingEmitter(
        mutation_fn=lambda x, y: x,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )

    metrics_fn = functools.partial(default_qd_metrics, qd_offset=2.0)

    mtme = MultiTaskMAPElites(
        scoring_function=multi_task_arm_scoring_function,
        emitter=emitter,
        metrics_function=metrics_fn,
        tournament_sizes=(1,),
    )

    key, subkey = jax.random.split(key)
    init_genotypes = jax.random.uniform(
        subkey, shape=(batch_size, num_param_dimensions)
    )
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, mtme_state, init_metrics = mtme.init(
        init_genotypes, task_descriptors, subkey
    )

    (
        repertoire,
        emitter_state,
        mtme_state,
        key,
    ), metrics = jax.lax.scan(
        mtme.scan_update,
        (repertoire, emitter_state, mtme_state, key),
        (),
        length=num_iterations,
    )

    pytest.assume(float(metrics["coverage"][-1]) > 0.0)


def test_mtme_tournament_requires_parent_descriptors() -> None:
    """A tournament larger than 1 requires the parent descriptors."""
    seed = 0
    num_param_dimensions = 8
    num_tasks = 32
    batch_size = 16

    key = jax.random.key(seed)

    key, subkey = jax.random.split(key)
    task_descriptors = get_task_descriptors(subkey, num_tasks)

    variation_fn = functools.partial(isoline_variation, iso_sigma=0.05, line_sigma=0.1)
    emitter = MixingEmitter(
        mutation_fn=lambda x, y: x,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )

    metrics_fn = functools.partial(default_qd_metrics, qd_offset=2.0)

    mtme = MultiTaskMAPElites(
        scoring_function=multi_task_arm_scoring_function,
        emitter=emitter,
        metrics_function=metrics_fn,
        tournament_sizes=(1, 10),
    )

    key, subkey = jax.random.split(key)
    init_genotypes = jax.random.uniform(
        subkey, shape=(batch_size, num_param_dimensions)
    )
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, mtme_state, init_metrics = mtme.init(
        init_genotypes, task_descriptors, subkey
    )

    key, subkey = jax.random.split(key)
    with pytest.raises(ValueError):
        mtme.update(repertoire, emitter_state, mtme_state, subkey)


def test_mtme_ask_tell() -> None:
    seed = 42
    num_param_dimensions = 10
    num_tasks = 64
    batch_size = 32
    num_iterations = 5

    key = jax.random.key(seed)

    key, subkey = jax.random.split(key)
    task_descriptors = get_task_descriptors(subkey, num_tasks)

    variation_fn = functools.partial(isoline_variation, iso_sigma=0.05, line_sigma=0.1)
    emitter = MultiTaskMixingEmitter(
        mutation_fn=lambda x, y: x,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )

    metrics_fn = functools.partial(default_qd_metrics, qd_offset=2.0)

    mtme = MultiTaskMAPElites(
        scoring_function=None,
        emitter=emitter,
        metrics_function=metrics_fn,
        tournament_sizes=(1, 5, 10),
    )

    # Evaluate the initial population
    key, subkey = jax.random.split(key)
    init_genotypes = jax.random.uniform(
        subkey, shape=(batch_size, num_param_dimensions)
    )
    key, subkey = jax.random.split(key)
    task_indices = jax.random.randint(subkey, (batch_size,), 0, num_tasks)
    key, subkey = jax.random.split(key)
    fitnesses, extra_scores = multi_task_arm_scoring_function(
        init_genotypes, task_descriptors[task_indices], subkey
    )

    repertoire, emitter_state, mtme_state, init_metrics = mtme.init_ask_tell(
        genotypes=init_genotypes,
        fitnesses=fitnesses,
        task_indices=task_indices,
        task_descriptors=task_descriptors,
        key=key,
        extra_scores=extra_scores,
    )

    ask_fn = jax.jit(mtme.ask)
    tell_fn = jax.jit(mtme.tell)

    for _ in range(num_iterations):
        key, subkey = jax.random.split(key)
        genotypes, task_indices, extra_info = ask_fn(
            repertoire, emitter_state, mtme_state, subkey
        )

        key, subkey = jax.random.split(key)
        fitnesses, extra_scores = multi_task_arm_scoring_function(
            genotypes, task_descriptors[task_indices], subkey
        )

        repertoire, emitter_state, mtme_state, current_metrics = tell_fn(
            genotypes=genotypes,
            fitnesses=fitnesses,
            task_indices=task_indices,
            repertoire=repertoire,
            emitter_state=emitter_state,
            mtme_state=mtme_state,
            extra_scores=extra_scores,
            extra_info=extra_info,
        )

    pytest.assume(repertoire is not None)
    pytest.assume(int(mtme_state.generation) == num_iterations)


if __name__ == "__main__":
    test_mtme(tournament_sizes=(1, 5, 10, 50))
    test_mtme_uniform_task_selection_with_standard_emitter()
    test_mtme_tournament_requires_parent_descriptors()
    test_mtme_ask_tell()
