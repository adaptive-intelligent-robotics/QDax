from functools import partial
from typing import Any, Tuple

import jax
import pytest
from brax.envs import State as EnvState

from qdax import environments
from qdax.baselines.td3 import TD3, TD3Config, TD3TrainingState
from qdax.core.neuroevolution.buffers.buffer import ReplayBuffer, Transition
from qdax.core.neuroevolution.sac_td3_utils import do_iteration_fn, warmstart_buffer


def test_td3() -> None:
    env_name = "pointmaze"
    seed = 0
    env_batch_size = 10
    num_steps = 10000
    warmup_steps = 1000
    buffer_size = 10000

    episode_length = 1000
    grad_updates_per_step = 1
    soft_tau_update = 0.005
    expl_noise = 0.1
    batch_size = 256
    policy_delay = 2
    discount = 0.95
    noise_clip = 0.5
    policy_noise = 0.2
    reward_scaling = 1.0
    critic_hidden_layer_size = (256, 256)
    policy_hidden_layer_size = (256, 256)
    critic_learning_rate = 3e-4
    policy_learning_rate = 3e-4

    # Create environment
    env = environments.create(
        env_name=env_name,
        batch_size=env_batch_size,
        episode_length=episode_length,
        auto_reset=True,
    )
    # Create eval environment
    eval_env = environments.create(
        env_name=env_name,
        batch_size=env_batch_size,
        episode_length=episode_length,
        auto_reset=True,
        eval_metrics=True,
    )
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    env_state = jax.jit(env.reset)(rng=key)
    eval_env_first_state = jax.jit(eval_env.reset)(rng=key)

    # Initialize buffer
    dummy_transition = Transition.init_dummy(
        observation_dim=env.observation_size, action_dim=env.action_size
    )
    replay_buffer = ReplayBuffer.init(
        buffer_size=buffer_size, transition=dummy_transition
    )

    td3_config = TD3Config(
        episode_length=episode_length,
        batch_size=batch_size,
        policy_delay=policy_delay,
        soft_tau_update=soft_tau_update,
        expl_noise=expl_noise,
        critic_hidden_layer_size=critic_hidden_layer_size,
        policy_hidden_layer_size=policy_hidden_layer_size,
        critic_learning_rate=critic_learning_rate,
        policy_learning_rate=policy_learning_rate,
        discount=discount,
        noise_clip=noise_clip,
        policy_noise=policy_noise,
        reward_scaling=reward_scaling,
    )

    # Initialize TD3 algorithm
    td3 = TD3(config=td3_config, action_size=env.action_size)

    key, subkey = jax.random.split(key)
    training_state = td3.init(
        key, action_size=env.action_size, observation_size=env.observation_size
    )

    # Wrap and jit play step function
    play_step = partial(
        td3.play_step_fn,
        env=env,
        deterministic=False,
    )

    # Wrap and jit play eval step function
    play_eval_step = partial(td3.play_step_fn, env=eval_env, deterministic=True)

    # Wrap and jit eval policy function
    eval_policy = partial(
        td3.eval_policy_fn,
        play_step_fn=play_eval_step,
        eval_env_first_state=eval_env_first_state,
    )

    # Wrap and jit do iteration function
    do_iteration = partial(
        do_iteration_fn,
        env_batch_size=env_batch_size,
        grad_updates_per_step=grad_updates_per_step,
        play_step_fn=play_step,
        update_fn=td3.update,
    )

    def _scan_do_iteration(
        carry: Tuple[TD3TrainingState, EnvState, ReplayBuffer],
        unused_arg: Any,
    ) -> Tuple[Tuple[TD3TrainingState, EnvState, ReplayBuffer], Any]:
        (
            training_state,
            env_state,
            replay_buffer,
            metrics,
        ) = do_iteration(*carry)
        return (training_state, env_state, replay_buffer), metrics

    # Warmstart the buffer
    replay_buffer, env_state, training_state = warmstart_buffer(
        replay_buffer=replay_buffer,
        training_state=training_state,
        env_state=env_state,
        num_warmstart_steps=warmup_steps,
        env_batch_size=env_batch_size,
        play_step_fn=play_step,
    )

    # Evaluate untrained policy
    true_return, true_returns = eval_policy(training_state=training_state)

    total_num_iterations = num_steps // env_batch_size

    # Main training loop: update agent, evaluate and log metrics
    (training_state, env_state, replay_buffer), metrics = jax.lax.scan(
        _scan_do_iteration,
        (training_state, env_state, replay_buffer),
        (),
        length=total_num_iterations,
    )

    # Evaluate
    final_true_return, final_true_returns = eval_policy(training_state=training_state)

    pytest.assume(final_true_return > true_return)


if __name__ == "__main__":
    test_td3()
