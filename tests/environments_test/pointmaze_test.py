from typing import Any, Tuple

import brax
import brax.envs
import jax
import pytest
from brax import jumpy as jp

import qdax
from qdax.environments.pointmaze import PointMaze
from qdax.types import EnvState


def test_pointmaze() -> None:
    # create env with class
    qd_env = PointMaze()
    # verify class
    pytest.assume(isinstance(qd_env, brax.envs.Env))

    # check state_descriptor_length
    pytest.assume(qd_env.state_descriptor_length == 2)
    pytest.assume(qd_env.state_descriptor_name == "xy_position")

    # create env with name
    qd_env = qdax.environments.create(env_name="pointmaze")  # type: ignore

    # verify class
    pytest.assume(isinstance(qd_env, brax.envs.Env))

    # check state_descriptor_length
    pytest.assume(qd_env.state_descriptor_length == 2)
    pytest.assume(qd_env.state_descriptor_name == "xy_position")

    # check that the classic functions are still working
    state = qd_env.reset(rng=jp.random_prngkey(seed=0))
    for _ in range(4):
        action = jp.zeros((qd_env.action_size,))
        state = qd_env.step(state, action)

        # check state size
        pytest.assume(state.obs.size == qd_env.observation_size)

        # check that the feet contact info exist
        pytest.assume("state_descriptor" in state.info.keys())

        # retrieve feet contact
        state_descriptor = state.info["state_descriptor"]

        # check that it has the good dimensions
        pytest.assume(len(state_descriptor) == qd_env.state_descriptor_length)

    # check coherence with the unwrapped original environment
    # create new envs
    qd_env = qdax.environments.create(env_name="pointmaze")  # type: ignore

    # test with jit - only jit the step
    state = qd_env.reset(rng=jp.random_prngkey(seed=0))
    for _ in range(4):
        action = jp.zeros((qd_env.action_size,))
        state = jax.jit(qd_env.step)(state, action)

    # test with jit - jit the entire rollout
    state = qd_env.reset(rng=jp.random_prngkey(seed=0))

    @jax.jit
    def run_n_steps(state: EnvState) -> EnvState:
        def run_step(carry: Tuple[EnvState], _: Any) -> Tuple[Tuple[EnvState], Any]:
            (state,) = carry
            action = jp.zeros((qd_env.action_size,))
            state = qd_env.step(state, action)
            return (state,), ()

        (final_state,), _ = jax.lax.scan(run_step, (state,), (), length=4)

        return final_state

    jax.jit(run_n_steps)(state)
