"""Testing script for the algorithm ME PBT SAC"""

import functools

import jax
import jax.numpy as jnp
import pytest

from qdax import environments
from qdax.baselines.sac_pbt import PBTSAC, PBTSacConfig
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.distributed_map_elites import DistributedMAPElites
from qdax.core.emitters.pbt_me_emitter import PBTEmitter, PBTEmitterConfig
from qdax.core.emitters.pbt_variation_operators import sac_pbt_variation_fn
from qdax.utils.metrics import default_qd_metrics


def test_me_pbt_sac() -> None:
    devices = jax.devices("cpu")
    num_devices = len(devices)

    env_name = "pointmaze"
    seed = 0

    # SAC config
    batch_size = 16
    episode_length = 100
    tau = 0.005
    alpha_init = 1.0
    policy_hidden_layer_size = (64, 64)
    critic_hidden_layer_size = (64, 64)
    fix_alpha = False
    normalize_observations = False

    # Emitter config
    buffer_size = 100000
    pg_population_size_per_device = 10
    ga_population_size_per_device = 30
    num_training_iterations = 100
    env_batch_size = 25
    grad_updates_per_step = 1.0
    iso_sigma = 0.005
    line_sigma = 0.05

    fraction_best_to_replace_from = 0.1
    fraction_to_replace_from_best = 0.2
    fraction_to_replace_from_samples = 0.4

    eval_env_batch_size = 1

    # MAP-Elites config
    num_init_cvt_samples = 50000
    num_centroids = 128
    num_loops = 10

    # Initialize environments
    env = environments.create(
        env_name=env_name,
        batch_size=env_batch_size * pg_population_size_per_device,
        episode_length=episode_length,
        auto_reset=True,
    )

    eval_env = environments.create(
        env_name=env_name,
        batch_size=eval_env_batch_size,
        episode_length=episode_length,
        auto_reset=True,
    )
    min_descriptor, max_descriptor = env.descriptor_limits

    key = jax.random.key(seed)

    key, subkey = jax.random.split(key)
    eval_env_first_states = jax.jit(eval_env.reset)(rng=subkey)

    # get agent
    config = PBTSacConfig(
        batch_size=batch_size,
        episode_length=episode_length,
        tau=tau,
        normalize_observations=normalize_observations,
        alpha_init=alpha_init,
        policy_hidden_layer_size=policy_hidden_layer_size,
        critic_hidden_layer_size=critic_hidden_layer_size,
        fix_alpha=fix_alpha,
    )

    agent = PBTSAC(config=config, action_size=env.action_size)

    # init emitter
    emitter_config = PBTEmitterConfig(
        buffer_size=buffer_size,
        num_training_iterations=num_training_iterations // env_batch_size,
        env_batch_size=env_batch_size,
        grad_updates_per_step=grad_updates_per_step,
        pg_population_size_per_device=pg_population_size_per_device,
        ga_population_size_per_device=ga_population_size_per_device,
        num_devices=num_devices,
        fraction_best_to_replace_from=fraction_best_to_replace_from,
        fraction_to_replace_from_best=fraction_to_replace_from_best,
        fraction_to_replace_from_samples=fraction_to_replace_from_samples,
        fraction_sort_exchange=0.1,
    )

    variation_fn = functools.partial(
        sac_pbt_variation_fn, iso_sigma=iso_sigma, line_sigma=line_sigma
    )

    emitter = PBTEmitter(
        pbt_agent=agent,
        config=emitter_config,
        env=env,
        variation_fn=variation_fn,
    )

    # get scoring function
    descriptor_extraction_fn = environments.descriptor_extractor[env_name]
    eval_policy = agent.get_eval_qd_fn(
        eval_env, descriptor_extraction_fn=descriptor_extraction_fn
    )

    def scoring_function(genotypes, key):  # type: ignore
        population_size = jax.tree.leaves(genotypes)[0].shape[0]
        first_states = jax.tree.map(
            lambda x: jnp.expand_dims(x, axis=0), eval_env_first_states
        )
        first_states = jax.tree.map(
            lambda x: jnp.repeat(x, population_size, axis=0), first_states
        )
        population_returns, population_descriptors, _, _ = eval_policy(
            genotypes, first_states
        )
        return population_returns, population_descriptors, {}

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name]

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * episode_length,
    )

    # Get the MAP-Elites algorithm
    map_elites = DistributedMAPElites(
        scoring_function=scoring_function,
        emitter=emitter,
        metrics_function=metrics_function,
    )

    key, subkey = jax.random.split(key)
    centroids = compute_cvt_centroids(
        num_descriptors=env.descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_descriptor,
        maxval=max_descriptor,
        key=subkey,
    )

    key, *keys = jax.random.split(key, num=1 + num_devices)
    keys = jnp.stack(keys)

    # get the initial training states and replay buffers
    agent_init_fn = agent.get_init_fn(
        population_size=pg_population_size_per_device + ga_population_size_per_device,
        action_size=env.action_size,
        observation_size=env.observation_size,
        buffer_size=buffer_size,
    )

    # Need to convert to PRNGKey because of github.com/jax-ml/jax/issues/23647
    keys = jax.random.key_data(keys)
    training_states, _ = jax.pmap(agent_init_fn, axis_name="p", devices=devices)(keys)

    # empty optimizers states to avoid too heavy repertories
    training_states = jax.pmap(
        jax.vmap(training_states.__class__.empty_optimizers_states),
        axis_name="p",
        devices=devices,
    )(training_states)

    # initialize map-elites
    repertoire, emitter_state, init_metrics = map_elites.get_distributed_init_fn(
        devices=devices, centroids=centroids
    )(
        genotypes=training_states, key=keys
    )  # type: ignore

    update_fn = map_elites.get_distributed_update_fn(num_iterations=1, devices=devices)

    initial_metrics = jax.pmap(metrics_function, axis_name="p", devices=devices)(
        repertoire
    )
    initial_metrics_cpu = jax.tree.map(
        lambda x: jax.device_put(x, jax.devices("cpu")[0])[0], initial_metrics
    )
    initial_qd_score = initial_metrics_cpu["qd_score"]

    for _ in range(num_loops):

        repertoire, emitter_state, metrics = update_fn(repertoire, emitter_state, keys)
        metrics_cpu = jax.tree.map(
            lambda x: jax.device_put(x, jax.devices("cpu")[0])[0], metrics
        )

    final_qd_score = metrics_cpu["qd_score"]

    pytest.assume(final_qd_score > initial_qd_score)
