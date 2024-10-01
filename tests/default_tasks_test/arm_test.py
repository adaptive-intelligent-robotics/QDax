"""Test default rastrigin using MAP Elites"""

import functools

import jax
import jax.numpy as jnp
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
        noisy_arm_scoring_function,
        fit_variance=0.1,
        desc_variance=0.1,
        params_variance=0.05,
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
    min_descriptor = 0.0
    max_descriptor = 1.0

    # Init a random key
    key = jax.random.key(seed)

    # Init population of controllers
    key, subkey = jax.random.split(key)
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


def test_arm_scoring_function() -> None:

    # Init a random key
    seed = 42
    key = jax.random.key(seed)

    # arm has xy descriptor centered at 0.5 0.5 and min max range is [0,1]
    # 0 params of first genotype is horizontal and points towards negative x axis
    # angles move in anticlockwise direction
    genotypes_1 = jnp.ones(shape=(1, 4)) * 0.5  # 0.5
    genotypes_2 = jnp.zeros(
        shape=(1, 6)
    )  # zeros - this folds upon itself (if even number ends up at origin)
    genotypes_3 = jnp.ones(
        shape=(1, 10)
    )  # ones - this also folds upon itself (if even number ends up at origin)
    genotypes_4 = jnp.array([[0, 0.5]])
    genotypes_5 = jnp.array([[0.25, 0.5]])
    genotypes_6 = jnp.array([[0.5, 0.5]])
    genotypes_7 = jnp.array([[0.75, 0.5]])

    fitness_1, descriptors_1, _ = arm_scoring_function(genotypes_1, key)
    fitness_2, descriptors_2, _ = arm_scoring_function(genotypes_2, key)
    fitness_3, descriptors_3, _ = arm_scoring_function(genotypes_3, key)
    fitness_4, descriptors_4, _ = arm_scoring_function(genotypes_4, key)
    fitness_5, descriptors_5, _ = arm_scoring_function(genotypes_5, key)
    fitness_6, descriptors_6, _ = arm_scoring_function(genotypes_6, key)
    fitness_7, descriptors_7, _ = arm_scoring_function(genotypes_7, key)

    # use rounding to avoid some numerical floating point errors
    pytest.assume(
        jnp.array_equal(jnp.around(descriptors_1, decimals=1), jnp.array([[1.0, 0.5]]))
    )
    pytest.assume(
        jnp.array_equal(jnp.around(descriptors_2, decimals=1), jnp.array([[0.5, 0.5]]))
    )
    pytest.assume(
        jnp.array_equal(jnp.around(descriptors_3, decimals=1), jnp.array([[0.5, 0.5]]))
    )
    pytest.assume(
        jnp.array_equal(jnp.around(descriptors_4, decimals=1), jnp.array([[0.0, 0.5]]))
    )
    pytest.assume(
        jnp.array_equal(jnp.around(descriptors_5, decimals=1), jnp.array([[0.5, 0.0]]))
    )
    pytest.assume(
        jnp.array_equal(jnp.around(descriptors_6, decimals=1), jnp.array([[1.0, 0.5]]))
    )
    pytest.assume(
        jnp.array_equal(jnp.around(descriptors_7, decimals=1), jnp.array([[0.5, 1.0]]))
    )


if __name__ == "__main__":
    test_arm(task_name="arm", batch_size=128)
    test_arm_scoring_function()
