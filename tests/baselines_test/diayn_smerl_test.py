"""Testing script for the algorithm DIAYN SMERL"""

from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import pytest
from brax.envs import State as EnvState

from qdax import environments
from qdax.baselines.diayn_smerl import DIAYNSMERL, DiaynSmerlConfig, DiaynTrainingState
from qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from qdax.core.neuroevolution.buffers.trajectory_buffer import TrajectoryBuffer
from qdax.core.neuroevolution.sac_td3_utils import do_iteration_fn, warmstart_buffer


def test_diayn_smerl() -> None:
    """Launches and monitors the training of the agent."""

    env_name = "pointmaze"
    seed = 0
    env_batch_size = 100
    num_steps = 10000
    warmup_steps = 0
    buffer_size = 10000

    # SAC config
    batch_size = 256
    episode_length = 100
    grad_updates_per_step = 0.1
    tau = 0.005
    learning_rate = 3e-4
    alpha_init = 1.0
    discount = 0.97
    reward_scaling = 1.0
    critic_hidden_layer_size: tuple = (256, 256)
    policy_hidden_layer_size: tuple = (256, 256)
    fix_alpha = False
    normalize_observations = False
    # DIAYN config
    num_skills = 5
    descriptor_full_state = False

    # SMERL specific
    diversity_reward_scale = 2.0
    smerl_target = -200
    smerl_margin = 40

    # Initialize environments
    env_batch_size = env_batch_size
    assert (
        env_batch_size % num_skills == 0
    ), "Parameter env_batch_size should be a multiple of num_skills"
    num_env_per_skill = env_batch_size // num_skills

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
    dummy_transition = QDTransition.init_dummy(
        observation_dim=env.observation_size + num_skills,
        action_dim=env.action_size,
        descriptor_dim=env.behavior_descriptor_length,
    )
    replay_buffer = TrajectoryBuffer.init(
        buffer_size=buffer_size,
        transition=dummy_transition,
        env_batch_size=env_batch_size,
        episode_length=episode_length,
    )

    if descriptor_full_state:
        descriptor_size = env.observation_size
    else:
        descriptor_size = env.behavior_descriptor_length

    diayn_smerl_config = DiaynSmerlConfig(
        # SAC config
        batch_size=batch_size,
        episode_length=episode_length,
        tau=tau,
        normalize_observations=normalize_observations,
        learning_rate=learning_rate,
        alpha_init=alpha_init,
        discount=discount,
        reward_scaling=reward_scaling,
        critic_hidden_layer_size=critic_hidden_layer_size,
        policy_hidden_layer_size=policy_hidden_layer_size,
        fix_alpha=fix_alpha,
        # DIAYN config
        num_skills=num_skills,
        descriptor_full_state=descriptor_full_state,
        # SMERL config
        diversity_reward_scale=diversity_reward_scale,
        smerl_margin=smerl_margin,
        smerl_target=smerl_target,
    )

    diayn_smerl = DIAYNSMERL(config=diayn_smerl_config, action_size=env.action_size)
    training_state = diayn_smerl.init(
        key,
        action_size=env.action_size,
        observation_size=env.observation_size,
        descriptor_size=descriptor_size,
    )

    skills = jnp.concatenate(
        [jnp.eye(num_skills)] * num_env_per_skill,
        axis=0,
    )

    # Make play_step* functions scannable by passing static args beforehand

    play_eval_step = partial(
        diayn_smerl.play_step_fn,
        skills=skills,
        env=eval_env,
        deterministic=True,
    )

    play_step = partial(
        diayn_smerl.play_step_fn,
        skills=skills,
        env=env,
        deterministic=False,
    )

    eval_policy = partial(
        diayn_smerl.eval_policy_fn,
        play_step_fn=play_eval_step,
        eval_env_first_state=eval_env_first_state,
        env_batch_size=env_batch_size,
    )

    # warmstart the buffer
    replay_buffer, env_state, training_state = warmstart_buffer(
        replay_buffer=replay_buffer,
        training_state=training_state,
        env_state=env_state,
        num_warmstart_steps=warmup_steps,
        env_batch_size=env_batch_size,
        play_step_fn=play_step,
    )

    total_num_iterations = num_steps // env_batch_size

    do_iteration = partial(
        do_iteration_fn,
        env_batch_size=env_batch_size,
        grad_updates_per_step=grad_updates_per_step,
        play_step_fn=play_step,
        update_fn=diayn_smerl.update,
    )

    @jax.jit
    def _scan_do_iteration(
        carry: Tuple[DiaynTrainingState, EnvState, ReplayBuffer],
        unused_arg: Any,
    ) -> Tuple[Tuple[DiaynTrainingState, EnvState, ReplayBuffer], Any]:
        (
            training_state,
            env_state,
            replay_buffer,
            metrics,
        ) = do_iteration(*carry)
        return (training_state, env_state, replay_buffer), metrics

    # Main loop
    (training_state, env_state, replay_buffer), metrics = jax.lax.scan(
        _scan_do_iteration,
        (training_state, env_state, replay_buffer),
        (),
        length=total_num_iterations,
    )

    # Evaluation part
    true_return, true_returns, diversity_returns, state_desc = eval_policy(
        training_state=training_state
    )

    pytest.assume(true_return > -200)


if __name__ == "__main__":
    test_diayn_smerl()
