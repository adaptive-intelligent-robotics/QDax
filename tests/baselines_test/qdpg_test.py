import functools
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import pytest

from qdax import environments
from qdax.core.containers.archive import score_euclidean_novelty
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from qdax.core.emitters.dpg_emitter import DiversityPGConfig
from qdax.core.emitters.qdpg_emitter import QDPGEmitter, QDPGEmitterConfig
from qdax.core.emitters.qpg_emitter import QualityPGConfig
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import EnvState, Params, RNGKey
from qdax.tasks.brax_envs import scoring_function_brax_envs as scoring_function


def test_qdpg() -> None:
    env_name = "walker2d_uni"
    episode_length = 100
    num_iterations = 5
    seed = 42
    policy_hidden_layer_sizes = (64, 64)
    num_init_cvt_samples = 1000
    num_centroids = 50
    min_descriptor = 0.0
    max_descriptor = 1.0

    # mutations size
    quality_pg_batch_size = 3
    diversity_pg_batch_size = 3
    ga_batch_size = 3

    env_batch_size = quality_pg_batch_size + diversity_pg_batch_size + ga_batch_size

    # TD3 params
    replay_buffer_size = 100000
    critic_hidden_layer_size = (64, 64)
    critic_learning_rate = 3e-4
    actor_learning_rate = 3e-4
    policy_learning_rate = 1e-3
    noise_clip = 0.5
    policy_noise = 0.2
    discount = 0.99
    reward_scaling = 1.0
    transitions_batch_size = 32
    soft_tau_update = 0.005
    num_critic_training_steps = 5
    num_pg_training_steps = 5
    policy_delay = 2

    # archive
    archive_acceptance_threshold = 0.1
    archive_max_size = 10000

    iso_sigma = 0.05
    line_sigma = 0.1

    num_nearest_neighb = 5
    novelty_scaling_ratio = 1.0

    # Init environment
    env = environments.create(env_name, episode_length=episode_length)
    reset_fn = jax.jit(env.reset)

    # Init a random key
    key = jax.random.key(seed)

    # Init policy network
    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of controllers
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=env_batch_size)
    fake_batch = jnp.zeros(shape=(env_batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # Define the function to play a step with the policy in the environment
    def play_step_fn(
        env_state: EnvState,
        policy_params: Params,
        key: RNGKey,
    ) -> Tuple[EnvState, Params, RNGKey, QDTransition]:
        """
        Play an environment step and return the updated state and the transition.
        """

        actions = policy_network.apply(policy_params, env_state.obs)

        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return next_state, policy_params, key, transition

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name]

    # Define a metrics function
    def metrics_function(repertoire: MapElitesRepertoire) -> Dict:

        # Get metrics
        grid_empty = repertoire.fitnesses == -jnp.inf
        qd_score = jnp.sum(repertoire.fitnesses, where=~grid_empty)
        # Add offset for positive qd_score
        qd_score += reward_offset * episode_length * jnp.sum(1.0 - grid_empty)
        coverage = 100 * jnp.mean(1.0 - grid_empty)
        max_fitness = jnp.max(repertoire.fitnesses)

        return {"qd_score": qd_score, "max_fitness": max_fitness, "coverage": coverage}

    # Define the Quality PG emitter config
    qpg_emitter_config = QualityPGConfig(
        env_batch_size=quality_pg_batch_size,
        batch_size=transitions_batch_size,
        critic_hidden_layer_size=critic_hidden_layer_size,
        critic_learning_rate=critic_learning_rate,
        actor_learning_rate=actor_learning_rate,
        policy_learning_rate=policy_learning_rate,
        noise_clip=noise_clip,
        policy_noise=policy_noise,
        discount=discount,
        reward_scaling=reward_scaling,
        replay_buffer_size=replay_buffer_size,
        soft_tau_update=soft_tau_update,
        num_critic_training_steps=num_critic_training_steps,
        num_pg_training_steps=num_pg_training_steps,
        policy_delay=policy_delay,
    )

    # Define the Diversity PG emitter config
    dpg_emitter_config = DiversityPGConfig(
        env_batch_size=diversity_pg_batch_size,
        batch_size=transitions_batch_size,
        critic_hidden_layer_size=critic_hidden_layer_size,
        critic_learning_rate=critic_learning_rate,
        actor_learning_rate=actor_learning_rate,
        policy_learning_rate=policy_learning_rate,
        noise_clip=noise_clip,
        policy_noise=policy_noise,
        discount=discount,
        reward_scaling=reward_scaling,
        replay_buffer_size=replay_buffer_size,
        soft_tau_update=soft_tau_update,
        num_critic_training_steps=num_critic_training_steps,
        num_pg_training_steps=num_pg_training_steps,
        policy_delay=policy_delay,
        archive_acceptance_threshold=archive_acceptance_threshold,
        archive_max_size=archive_max_size,
    )

    # Define the QDPG Emitter config
    qdpg_emitter_config = QDPGEmitterConfig(
        qpg_config=qpg_emitter_config,
        dpg_config=dpg_emitter_config,
        iso_sigma=iso_sigma,
        line_sigma=line_sigma,
        ga_batch_size=ga_batch_size,
    )

    score_novelty = jax.jit(
        functools.partial(
            score_euclidean_novelty,
            num_nearest_neighb=num_nearest_neighb,
            scaling_ratio=novelty_scaling_ratio,
        )
    )

    # define the QDPG emitter
    qdpg_emitter = QDPGEmitter(
        config=qdpg_emitter_config,
        policy_network=policy_network,
        env=env,
        score_novelty=score_novelty,
    )

    # Prepare the scoring function
    descriptor_extraction_fn = environments.descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function,
        episode_length=episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
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

    # Instantiate MAP Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=qdpg_emitter,
        metrics_function=metrics_function,
    )

    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = map_elites.init(
        init_variables, centroids, subkey
    )

    @jax.jit
    def update_scan_fn(carry: Any, _: Any) -> Any:
        # iterate over grid
        repertoire, emitter_state, key = carry
        key, subkey = jax.random.split(key)
        repertoire, emitter_state, metrics = map_elites.update(
            repertoire, emitter_state, subkey
        )
        return (repertoire, emitter_state, key), metrics

    # Run the algorithm
    (
        repertoire,
        emitter_state,
        key,
    ), _metrics = jax.lax.scan(
        update_scan_fn,
        (repertoire, emitter_state, key),
        (),
        length=num_iterations,
    )

    pytest.assume(repertoire is not None)
