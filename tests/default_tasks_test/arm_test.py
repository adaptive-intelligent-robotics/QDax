"""Test default rastrigin using MAP Elites"""

import functools

import jax
import pytest

from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.tasks.arm import arm_scoring_function, noisy_arm_scoring_function
from qdax.utils.metrics import default_qd_metrics

scoring_functions = {
    "arm": functools.partial(arm_scoring_function),
    "noisy_arm": functools.partial(
        noisy_arm_scoring_function, fit_variance=0.1, desc_variance=0.1
    ),
}


@pytest.mark.parametrize(
    "task_name, batch_size",
    [("arm", 1), ("noisy_arm", 10)],
)
def test_arm(task_name: str, batch_size: int) -> None:
    seed = 42
    num_param_dimensions = 100  # num DoF arm
    init_batch_size = 100
    batch_size = batch_size
    num_iterations = 5
    grid_shape = (100, 100)
    min_param = 0.0
    max_param = 1.0
    min_bd = 0.0
    max_bd = 1.0

    # Init a random key
    random_key = jax.random.PRNGKey(seed)

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    init_variables = jax.random.uniform(
        subkey,
        shape=(init_batch_size, num_param_dimensions),
        minval=min_param,
        maxval=max_param,
    )

    # Prepare the scoring function
    scoring_fn = scoring_functions[task_name]

    # Define emitter
    variation_fn = functools.partial(
        isoline_variation,
        iso_sigma=0.05,
        line_sigma=0.1,
        minval=min_param,
        maxval=max_param,
    )
    mixing_emitter = MixingEmitter(
        mutation_fn=lambda x, y: (x, y),
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )

    # Define a metrics function
    metrics_fn = functools.partial(
        default_qd_metrics,
        qd_offset=0.0,
    )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_fn,
    )

    # Compute the centroids
    centroids = compute_euclidean_centroids(
        grid_shape=grid_shape,
        minval=min_bd,
        maxval=max_bd,
    )

    # Compute initial repertoire
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )

    # Run the algorithm
    (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
        map_elites.scan_update,
        (repertoire, emitter_state, random_key),
        (),
        length=num_iterations,
    )

    pytest.assume(repertoire is not None)


if __name__ == "__main__":
    test_arm(task_name="arm", batch_size=128)
