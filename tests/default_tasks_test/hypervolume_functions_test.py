"""Test default rastrigin using MAP Elites"""

import functools

import jax
import pytest

from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.tasks.hypervolume_functions import (
    checkered_scoring_function,
    continous_islands_scoring_function,
    empty_circle_scoring_function,
    non_continous_islands_scoring_function,
    square_scoring_function,
)
from qdax.utils.metrics import default_qd_metrics

scoring_functions = {
    "square": square_scoring_function,
    "checkered": checkered_scoring_function,
    "empty_circle": empty_circle_scoring_function,
    "non_continous_islands": non_continous_islands_scoring_function,
    "continous_islands": continous_islands_scoring_function,
}


@pytest.mark.parametrize(
    "task_name, batch_size",
    [
        ("square", 1),
        ("checkered", 10),
        ("empty_circle", 20),
        ("non_continous_islands", 30),
        ("continous_islands", 40),
    ],
)
def test_standard_functions(task_name: str, batch_size: int) -> None:
    seed = 42
    num_param_dimensions = 2
    init_batch_size = 100
    batch_size = batch_size
    num_iterations = 5
    grid_shape = (100, 100)
    min_param = 0.0
    max_param = 1.0
    min_descriptor = 0.0
    max_descriptor = 1.0

    # Init a random key
    key = jax.random.key(seed)

    # Init population of controllers
    key, subkey = jax.random.split(key)
    init_variables = jax.random.uniform(
        subkey, shape=(init_batch_size, num_param_dimensions)
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
        minval=min_descriptor,
        maxval=max_descriptor,
    )

    # Compute initial repertoire
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = map_elites.init(
        init_variables, centroids, subkey
    )

    # Run the algorithm
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

    pytest.assume(repertoire is not None)


if __name__ == "__main__":
    test_standard_functions(task_name="rastrigin", batch_size=128)
