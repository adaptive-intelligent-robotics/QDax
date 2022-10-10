from typing import Any, Tuple

import brax
import jax
import jax.numpy as jnp
import pytest
from brax import jumpy as jp

import qdax
from qdax import environments

@pytest.mark.parametrize(
    "env_name",
    ["walker2d_uni", "ant_uni", "hopper_uni", "humanoid_uni", "halfcheetah_uni"],
)
def test_wrapper(env_name: str) -> None:
    """Test the wrapper running."""
    seed = 10

    # Init environment
    env = environments.create(env_name, fixed_init_state=True)
    print("Observation size: ",env.observation_size)
    print("Action size: ",env.action_size)

    random_key = jax.random.PRNGKey(seed)
    init_state = env.reset(random_key)

    joint_angle, joint_vel = env.sys.joints[0].angle_vel(init_state.qp)
    
    # check position and velocity
    pytest.assume(jnp.array_equal(joint_angle, env.sys.default_angle()))
    pytest.assume(jnp.array_equal(joint_vel, jp.zeros((env.sys.num_joint_dof,))))