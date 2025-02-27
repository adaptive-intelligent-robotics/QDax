"""Tests MAP Elites implementation"""

import functools

import jax
import pytest

import qdax.environments
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.mdp_utils import init_population_controllers
from qdax.tasks.brax_envs import create_default_brax_task_components
from qdax.utils.metrics import default_qd_metrics


@pytest.mark.parametrize(
    "env_name, batch_size, is_task_reset_based",
    [
        ("walker2d_uni", 5, False),
        ("walker2d_uni", 5, True),
    ],
)
def test_map_elites(env_name: str, batch_size: int, is_task_reset_based: bool) -> None:
    batch_size = batch_size
    env_name = env_name
    episode_length = 100
    num_iterations = 5
    seed = 42
    num_init_cvt_samples = 1000
    num_centroids = 50
    min_descriptor = 0.0
    max_descriptor = 1.0

    # Init a random key
    key = jax.random.key(seed)

    key, subkey = jax.random.split(key)
    env, policy_network, scoring_fn = create_default_brax_task_components(
        env_name=env_name,
        key=subkey,
    )

    # Define emitter
    variation_fn = functools.partial(isoline_variation, iso_sigma=0.05, line_sigma=0.1)
    mixing_emitter = MixingEmitter(
        mutation_fn=lambda x, y: (x, y),
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )

    # Define metrics function
    metrics_fn = functools.partial(
        default_qd_metrics,
        qd_offset=qdax.environments.reward_offset[env_name] * episode_length,
    )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_fn,
    )

    # Compute the centroids
    key, subkey = jax.random.split(key)
    centroids = compute_cvt_centroids(
        num_descriptors=env.descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_descriptor,
        maxval=max_descriptor,
        key=subkey,
    )

    # Init population of controllers
    key, subkey = jax.random.split(key)
    init_variables = init_population_controllers(
        policy_network, env, batch_size, subkey
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
    test_map_elites(env_name="walker2d_uni", batch_size=10, is_task_reset_based=False)
