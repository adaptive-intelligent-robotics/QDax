"""Testing script for the algorithm PBT SAC"""

import jax
import jax.numpy as jnp
import pytest

from qdax import environments
from qdax.baselines.pbt import PBT
from qdax.baselines.sac_pbt import PBTSAC, PBTSacConfig


def test_pbt_sac() -> None:
    devices = jax.devices("cpu")
    num_devices = len(devices)

    env_name = "pointmaze"
    seed = 0
    env_batch_size = 25
    population_size_per_device = 10
    population_size = population_size_per_device * num_devices
    num_steps = 100
    buffer_size = 100000

    # PBT Config
    num_best_to_replace_from = int(0.2 * population_size)
    num_worse_to_replace = int(0.4 * population_size)

    # SAC config
    batch_size = 16
    episode_length = 100
    grad_updates_per_step = 1.0
    tau = 0.005
    alpha_init = 1.0
    policy_hidden_layer_size = (64, 64)
    critic_hidden_layer_size = (64, 64)
    fix_alpha = False
    normalize_observations = False

    num_loops = 10

    # Initialize environments
    env = environments.create(
        env_name=env_name,
        batch_size=env_batch_size * population_size_per_device,
        episode_length=episode_length,
        auto_reset=True,
    )

    eval_env = environments.create(
        env_name=env_name,
        batch_size=env_batch_size * population_size_per_device,
        episode_length=episode_length,
        auto_reset=True,
    )

    @jax.jit
    def init_environments(random_key):  # type: ignore
        env_states = jax.jit(env.reset)(rng=random_key)
        eval_env_first_states = jax.jit(eval_env.reset)(rng=random_key)

        reshape_fn = jax.jit(
            lambda tree: jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x,
                    (
                        population_size_per_device,
                        env_batch_size,
                    )
                    + x.shape[1:],
                ),
                tree,
            ),
        )
        env_states = reshape_fn(env_states)
        eval_env_first_states = reshape_fn(eval_env_first_states)

        return env_states, eval_env_first_states

    key = jax.random.PRNGKey(seed)
    key, *keys = jax.random.split(key, num=1 + num_devices)
    keys = jnp.stack(keys)
    env_states, eval_env_first_states = jax.pmap(
        init_environments, axis_name="p", devices=devices
    )(keys)

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

    # get the initial training states and replay buffers
    agent_init_fn = agent.get_init_fn(
        population_size=population_size_per_device,
        action_size=env.action_size,
        observation_size=env.observation_size,
        buffer_size=buffer_size,
    )
    keys, training_states, replay_buffers = jax.pmap(
        agent_init_fn, axis_name="p", devices=devices
    )(keys)

    # get eval policy function
    eval_policy = jax.pmap(agent.get_eval_fn(eval_env), axis_name="p", devices=devices)

    # Evaluate untrained policy
    # eval policy before training
    population_returns, _ = eval_policy(training_states, eval_env_first_states)
    population_returns = jnp.reshape(population_returns, (population_size,))
    true_return = jnp.mean(population_returns)

    # get training function
    num_iterations = num_steps // env_batch_size

    train_fn = agent.get_train_fn(
        env=env,
        num_iterations=num_iterations,
        env_batch_size=env_batch_size,
        grad_updates_per_step=grad_updates_per_step,
    )
    train_fn = jax.pmap(train_fn, axis_name="p", devices=devices)

    pbt = PBT(
        population_size=population_size,
        num_best_to_replace_from=num_best_to_replace_from // num_devices,
        num_worse_to_replace=num_worse_to_replace // num_devices,
    )
    select_fn = jax.pmap(
        pbt.update_states_and_buffer_pmap, axis_name="p", devices=devices
    )

    for i in range(num_loops):

        # Update for num_steps
        (training_states, env_states, replay_buffers), metrics = train_fn(
            training_states, env_states, replay_buffers
        )

        # Eval policy after training
        population_returns, _ = eval_policy(training_states, eval_env_first_states)

        # PBT selection
        if i < (num_loops - 1):
            keys, training_states, replay_buffers = select_fn(
                keys, population_returns, training_states, replay_buffers
            )

    # Policy evaluation
    final_population_returns = jnp.reshape(population_returns, (population_size,))
    final_true_return = jnp.mean(final_population_returns)

    pytest.assume(final_true_return > true_return)
