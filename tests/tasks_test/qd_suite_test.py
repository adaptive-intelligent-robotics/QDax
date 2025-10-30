"""Test qd suite tasks using MAP Elites"""

import functools
import math
from typing import Tuple

import jax
import pytest

from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.tasks.qd_suite import (
    archimedean_spiral_v0_angle_euclidean_task,
    archimedean_spiral_v0_angle_geodesic_task,
    archimedean_spiral_v0_arc_length_euclidean_task,
    archimedean_spiral_v0_arc_length_geodesic_task,
    deceptive_evolvability_v0_task,
    ssf_v0_param_size_1_task,
    ssf_v0_param_size_2_task,
)
from qdax.utils.metrics import default_qd_metrics

task_dict = {
    "archimedean_spiral_v0_angle_euclidean": archimedean_spiral_v0_angle_euclidean_task,
    "archimedean_spiral_v0_angle_geodesic": archimedean_spiral_v0_angle_geodesic_task,
    "archimedean_spiral_v0_arc_length_euclidean": archimedean_spiral_v0_arc_length_euclidean_task,  # noqa: E501
    "archimedean_spiral_v0_arc_length_geodesic": archimedean_spiral_v0_arc_length_geodesic_task,  # noqa: E501
    "deceptive_evolvability_v0": deceptive_evolvability_v0_task,
    "ssf_v0_param_size_1": ssf_v0_param_size_1_task,
    "ssf_v0_param_size_2": ssf_v0_param_size_2_task,
}


@pytest.mark.parametrize(
    "task_name, batch_size",
    [
        ("archimedean_spiral_v0_angle_euclidean", 1),
        ("archimedean_spiral_v0_angle_geodesic", 10),
        ("archimedean_spiral_v0_arc_length_euclidean", 128),
        ("archimedean_spiral_v0_arc_length_geodesic", 30),
        ("deceptive_evolvability_v0", 64),
        ("ssf_v0_param_size_1", 256),
        ("ssf_v0_param_size_2", 1),
    ],
)
def test_qd_suite(task_name: str, batch_size: int) -> None:
    seed = 42

    # get task from parameterization for test
    task = task_dict[task_name]

    init_batch_size = 100
    batch_size = batch_size
    num_iterations = 5
    min_param, max_param = task.get_min_max_params()
    min_descriptor, max_descriptor = task.get_bounded_min_max_descriptor()
    descriptor_size = task.get_descriptor_size()

    grid_shape: Tuple[int, ...]
    if descriptor_size == 1:
        grid_shape = (100,)
    elif descriptor_size == 2:
        grid_shape = (100, 100)
    else:
        resolution_per_axis = math.floor(math.pow(10000.0, 1.0 / descriptor_size))
        grid_shape = tuple([resolution_per_axis for _ in range(descriptor_size)])

    # Init a random key
    key = jax.random.key(seed)

    # Init population of parameters
    init_variables = task.get_initial_parameters(init_batch_size)

    # Define scoring function
    scoring_fn = task.scoring_function

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
    test_qd_suite(task_name="archimedean_spiral_v0_angle_geodesic", batch_size=128)
