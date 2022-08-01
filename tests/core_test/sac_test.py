"""Testing script for the algorithm SAC"""

from functools import partial
from typing import Any, Tuple

import jax
import pytest

from qdax import environments
from qdax.baselines.sac import SAC, SacConfig, TrainingState
from qdax.core.neuroevolution.buffers.buffer import ReplayBuffer, Transition
from qdax.core.neuroevolution.sac_utils import do_iteration_fn, warmstart_buffer
from qdax.types import EnvState


def test_sac() -> None:
    env_name = "pointmaze"
    env_batch_size = 128
    seed = 0
    num_steps = 10000
    buffer_size = 10000
    warmup_steps = 0

    # SAC config
    batch_size = 512
    episode_length = 100
    grad_updates_per_step = 0.1
    tau = 0.005
    normalize_observations = False
    learning_rate = 6e-4
    alpha_init = 1.0
    discount = 0.95
    reward_scaling = 10.0
    hidden_layer_sizes = (64, 64)
    fix_alpha = False

    # Initialize environments
    env_batch_size = env_batch_size

    env = environments.create(
        env_name=env_name,
        batch_size=env_batch_size,
        episode_length=episode_length,
        auto_reset=True,
    )

    eval_env = environments.create(
        env_name=env_name,
        batch_size=env_batch_size,
        episode_length=episode_length,
        auto_reset=True,
        eval_metrics=True,
    )

    key = jax.random.PRNGKey(seed)
    env_state = jax.jit(env.reset)(rng=key)
    eval_env_first_state = jax.jit(eval_env.reset)(rng=key)

    # Initialize buffer
    dummy_transition = Transition.init_dummy(
        observation_dim=env.observation_size, action_dim=env.action_size
    )
    replay_buffer = ReplayBuffer.init(
        buffer_size=buffer_size, transition=dummy_transition
    )

    sac_config = SacConfig(
        batch_size=batch_size,
        episode_length=episode_length,
        grad_updates_per_step=grad_updates_per_step,
        tau=tau,
        normalize_observations=normalize_observations,
        learning_rate=learning_rate,
        alpha_init=alpha_init,
        discount=discount,
        reward_scaling=reward_scaling,
        hidden_layer_sizes=hidden_layer_sizes,
        fix_alpha=fix_alpha,
    )

    sac = SAC(config=sac_config, action_size=env.action_size)
    key, subkey = jax.random.split(key)
    training_state = sac.init(
        random_key=subkey,
        action_size=env.action_size,
        observation_size=env.observation_size,
    )

    # Make play_step* functions scannable by passing static args beforehand
    play_eval_step = partial(
        sac.play_step_fn, env=eval_env, deterministic=True, evaluation=True
    )

    play_step = partial(
        sac.play_step_fn,
        env=env,
        deterministic=False,
    )

    eval_policy = partial(
        sac.eval_policy_fn,
        play_step_fn=play_eval_step,
        eval_env_first_state=eval_env_first_state,
    )

    # warmstart the buffer
    key, subkey = jax.random.split(key)
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

    do_iteration = partial(
        do_iteration_fn,
        env_batch_size=env_batch_size,
        grad_updates_per_step=grad_updates_per_step,
        play_step_fn=play_step,
        update_fn=sac.update,
    )

    @jax.jit
    def _scan_do_iteration(
        carry: Tuple[TrainingState, EnvState, ReplayBuffer],
        unused_arg: Any,
    ) -> Tuple[Tuple[TrainingState, EnvState, ReplayBuffer], Any]:
        (
            training_state,
            env_state,
            replay_buffer,
            metrics,
        ) = do_iteration(*carry)
        return (training_state, env_state, replay_buffer), metrics

    # Training part
    (training_state, env_state, replay_buffer), (metrics) = jax.lax.scan(
        _scan_do_iteration,
        (training_state, env_state, replay_buffer),
        (),
        length=total_num_iterations,
    )

    # Evaluation
    # Policy evaluation
    final_true_return, final_true_returns = eval_policy(training_state=training_state)

    pytest.assume(final_true_return > true_return)


if __name__ == "__main__":
    test_sac()
