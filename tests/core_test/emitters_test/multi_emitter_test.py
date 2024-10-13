"""Test of Multiple Emitters"""

import functools

import jax
import pytest

from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.multi_emitter import MultiEmitter
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.tasks.standard_functions import rastrigin_scoring_function
from qdax.utils.metrics import default_qd_metrics


def test_multi_emitter() -> None:
    seed = 42
    num_param_dimensions = 100
    init_batch_size = 100
    batch_size = 128
    num_iterations = 5
    grid_shape = (100, 100)
    min_param = 0.0
    max_param = 1.0
    min_descriptor = min_param
    max_descriptor = max_param

    # Init a random key
    key = jax.random.key(seed)

    # Init population of controllers
    key, subkey = jax.random.split(key)
    init_variables = jax.random.uniform(
        subkey, shape=(init_batch_size, num_param_dimensions)
    )

    # Prepare the scoring function
    scoring_fn = rastrigin_scoring_function

    # Define emitter
    variation_fn_1 = functools.partial(
        isoline_variation,
        iso_sigma=0.05,
        line_sigma=0.0,
        minval=min_param,
        maxval=max_param,
    )
    variation_fn_2 = functools.partial(
        isoline_variation,
        iso_sigma=0.0,
        line_sigma=0.1,
        minval=min_param,
        maxval=max_param,
    )
    mixing_emitter_1 = MixingEmitter(
        mutation_fn=lambda x, y: (x, y),
        variation_fn=variation_fn_1,
        variation_percentage=1.0,
        batch_size=batch_size,
    )
    mixing_emitter_2 = MixingEmitter(
        mutation_fn=lambda x, y: (x, y),
        variation_fn=variation_fn_2,
        variation_percentage=1.0,
        batch_size=batch_size * 2,
    )

    emitter = MultiEmitter(
        emitters=(mixing_emitter_1, mixing_emitter_2),
    )

    # Define a metrics function
    metrics_fn = functools.partial(
        default_qd_metrics,
        qd_offset=0.0,
    )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=emitter,
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
    test_multi_emitter()
