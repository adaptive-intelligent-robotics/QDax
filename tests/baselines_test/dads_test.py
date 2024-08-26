"""Training script for the algorithm DADS, should be launched with hydra.
    e.g. python train_dads.py config=dads_ant"""

from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import pytest
from brax.envs import State as EnvState

from qdax import environments
from qdax.baselines.dads import DADS, DadsConfig, DadsTrainingState
from qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from qdax.core.neuroevolution.sac_td3_utils import do_iteration_fn, warmstart_buffer


def test_dads() -> None:
    """Launches and monitors the training of the agent."""

    env_name = "ant_omni"
    seed = 0
    env_batch_size = 200
    num_steps = 10000
    warmup_steps = 0
    buffer_size = 10000

    # SAC config
    batch_size = 256
    episode_length = 200
    tau = 0.005
    grad_updates_per_step = 0.25
    normalize_observations = False
    critic_hidden_layer_size: tuple = (256, 256)
    policy_hidden_layer_size: tuple = (256, 256)
    alpha_init = 1.0
    fix_alpha = False
    discount = 0.97
    reward_scaling = 1.0
    learning_rate = 3e-4
    # DADS config
    num_skills = 5
    dynamics_update_freq = 1
    normalize_target = True
    descriptor_full_state = True

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
    replay_buffer = ReplayBuffer.init(
        buffer_size=buffer_size, transition=dummy_transition
    )

    if descriptor_full_state:
        descriptor_size = env.observation_size
    else:
        descriptor_size = env.behavior_descriptor_length

    dads_config = DadsConfig(
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
        # DADS config
        num_skills=num_skills,
        descriptor_full_state=descriptor_full_state,
        omit_input_dynamics_dim=env.behavior_descriptor_length,
        dynamics_update_freq=dynamics_update_freq,
        normalize_target=normalize_target,
    )
    dads = DADS(
        config=dads_config,
        action_size=env.action_size,
        descriptor_size=descriptor_size,
    )
    training_state = dads.init(
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
        dads.play_step_fn,
        deterministic=True,
        env=eval_env,
        skills=skills,
        evaluation=True,
    )

    play_step = partial(
        dads.play_step_fn,
        env=env,
        deterministic=False,
        skills=skills,
    )

    eval_policy = partial(
        dads.eval_policy_fn,
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
        update_fn=dads.update,
    )

    @jax.jit
    def _scan_do_iteration(
        carry: Tuple[DadsTrainingState, EnvState, ReplayBuffer],
        unused_arg: Any,
    ) -> Tuple[Tuple[DadsTrainingState, EnvState, ReplayBuffer], Any]:
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

    # Evaluation part
    # Policy evaluation
    true_return, true_returns, diversity_returns, state_desc = eval_policy(
        training_state=training_state
    )

    print("True return : ", true_return)
    pytest.assume(true_return is not None)


if __name__ == "__main__":
    test_dads()
