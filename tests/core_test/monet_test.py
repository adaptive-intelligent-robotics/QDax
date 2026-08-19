"""Tests MONET implementation"""

import functools

import jax
import jax.numpy as jnp
import pytest

from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.monet import MONET
from qdax.tasks.arm import multi_task_arm_scoring_function
from qdax.utils.metrics import default_qd_metrics


def get_task_descriptors(key: jax.Array, num_tasks: int) -> jax.Array:
    """Spread the tasks in the (link_length, max_angle) space with a CVT."""
    return compute_cvt_centroids(
        num_descriptors=2,
        num_init_cvt_samples=1000,
        num_centroids=num_tasks,
        minval=0.0,
        maxval=1.0,
        key=key,
    )


@pytest.mark.parametrize(
    "neighbor_strategy",
    ["best_fitness", "random", "most_similar", "least_similar", "fitness_proportional"],
)
def test_monet(neighbor_strategy: str) -> None:
    seed = 42
    num_param_dimensions = 10
    num_tasks = 64
    batch_size = 32
    num_iterations = 10

    key = jax.random.key(seed)

    key, subkey = jax.random.split(key)
    task_descriptors = get_task_descriptors(subkey, num_tasks)

    metrics_fn = functools.partial(default_qd_metrics, qd_offset=2.0)

    scoring_fn = functools.partial(multi_task_arm_scoring_function, target=(0.5, 0.5))

    monet = MONET(
        scoring_function=scoring_fn,
        metrics_function=metrics_fn,
        batch_size=batch_size,
        num_neighbors=8,
        neighbor_strategy=neighbor_strategy,
        individual_learning=("gaussian_mutation", "polynomial_mutation"),
        social_learning=("sbx", "iso_dd"),
    )

    # Compute initial repertoire
    key, subkey = jax.random.split(key)
    init_genotypes = jax.random.uniform(
        subkey, shape=(batch_size, num_param_dimensions)
    )
    key, subkey = jax.random.split(key)
    repertoire, monet_state, init_metrics = monet.init(
        init_genotypes, task_descriptors, subkey
    )

    # the similarity graph has the expected shape
    pytest.assume(monet_state.neighbor_indices.shape == (num_tasks, 8))
    # no self-loops in the graph
    pytest.assume(
        bool(jnp.all(monet_state.neighbor_indices != jnp.arange(num_tasks)[:, None]))
    )

    # Run the algorithm
    (
        repertoire,
        monet_state,
        key,
    ), metrics = jax.lax.scan(
        monet.scan_update,
        (repertoire, monet_state, key),
        (),
        length=num_iterations,
    )

    pytest.assume(repertoire is not None)

    # some cells have been filled
    pytest.assume(float(metrics["coverage"][-1]) > 0.0)

    # genotypes remain in bounds
    pytest.assume(
        bool(jnp.all((repertoire.genotypes >= 0.0) & (repertoire.genotypes <= 1.0)))
    )

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


def test_monet_improves_mean_fitness() -> None:
    seed = 0
    num_param_dimensions = 10
    num_tasks = 100
    batch_size = 64
    num_iterations = 300

    key = jax.random.key(seed)

    key, subkey = jax.random.split(key)
    task_descriptors = get_task_descriptors(subkey, num_tasks)

    metrics_fn = functools.partial(default_qd_metrics, qd_offset=2.0)

    scoring_fn = functools.partial(multi_task_arm_scoring_function, target=(0.5, 0.5))

    monet = MONET(
        scoring_function=scoring_fn,
        metrics_function=metrics_fn,
        batch_size=batch_size,
        num_neighbors=max(1, num_tasks // 100),
    )

    key, subkey = jax.random.split(key)
    init_genotypes = jax.random.uniform(
        subkey, shape=(batch_size, num_param_dimensions)
    )
    key, subkey = jax.random.split(key)
    repertoire, monet_state, init_metrics = monet.init(
        init_genotypes, task_descriptors, subkey
    )

    (
        repertoire,
        monet_state,
        key,
    ), metrics = jax.lax.scan(
        monet.scan_update,
        (repertoire, monet_state, key),
        (),
        length=num_iterations,
    )

    # all tasks are covered and the qd score improved over the run
    pytest.assume(float(metrics["coverage"][-1]) == 100.0)
    pytest.assume(float(metrics["qd_score"][-1]) > float(metrics["qd_score"][0]))


def test_monet_invalid_configurations() -> None:
    metrics_fn = functools.partial(default_qd_metrics, qd_offset=2.0)

    with pytest.raises(ValueError):
        MONET(
            scoring_function=multi_task_arm_scoring_function,
            metrics_function=metrics_fn,
            batch_size=8,
            num_neighbors=4,
            individual_learning=("regression",),
        )
    with pytest.raises(ValueError):
        MONET(
            scoring_function=multi_task_arm_scoring_function,
            metrics_function=metrics_fn,
            batch_size=8,
            num_neighbors=4,
            social_learning=("plane_dd",),
        )
    with pytest.raises(ValueError):
        MONET(
            scoring_function=multi_task_arm_scoring_function,
            metrics_function=metrics_fn,
            batch_size=8,
            num_neighbors=4,
            neighbor_strategy="plane_dd",
        )
    with pytest.raises(ValueError):
        MONET(
            scoring_function=multi_task_arm_scoring_function,
            metrics_function=metrics_fn,
            batch_size=8,
            num_neighbors=4,
            init_strategy="unknown",
        )


def test_monet_ask_tell() -> None:
    seed = 42
    num_param_dimensions = 10
    num_tasks = 64
    batch_size = 32
    num_iterations = 5

    key = jax.random.key(seed)

    key, subkey = jax.random.split(key)
    task_descriptors = get_task_descriptors(subkey, num_tasks)

    metrics_fn = functools.partial(default_qd_metrics, qd_offset=2.0)

    monet = MONET(
        scoring_function=None,
        metrics_function=metrics_fn,
        batch_size=batch_size,
        num_neighbors=8,
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

    repertoire, monet_state, init_metrics = monet.init_ask_tell(
        genotypes=init_genotypes,
        fitnesses=fitnesses,
        task_indices=task_indices,
        task_descriptors=task_descriptors,
        key=key,
    )

    ask_fn = jax.jit(monet.ask)
    tell_fn = jax.jit(monet.tell)

    for _ in range(num_iterations):
        key, subkey = jax.random.split(key)
        genotypes, task_indices, extra_info = ask_fn(repertoire, monet_state, subkey)

        key, subkey = jax.random.split(key)
        fitnesses, extra_scores = multi_task_arm_scoring_function(
            genotypes, task_descriptors[task_indices], subkey
        )

        repertoire, monet_state, current_metrics = tell_fn(
            genotypes=genotypes,
            fitnesses=fitnesses,
            task_indices=task_indices,
            repertoire=repertoire,
            monet_state=monet_state,
            extra_scores=extra_scores,
        )

    pytest.assume(repertoire is not None)


if __name__ == "__main__":
    test_monet(neighbor_strategy="best_fitness")
    test_monet_improves_mean_fitness()
    test_monet_invalid_configurations()
    test_monet_ask_tell()
