"""Trains a hopper to run in the +x direction."""

from typing import Optional, Tuple

import brax
from brax import jumpy as jp
from brax.envs import env as brax_env
from brax.physics import bodies
from brax.envs import hopper

class DetHopper(hopper.Hopper):
  """Trains a hopper to run in the +x direction.
  This is similar to the Hopper-V3 Mujoco environment in OpenAI Gym.
  """

  def reset(self, rng: jp.ndarray) -> brax_env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jp.random_split(rng, 3)
    qpos = self.sys.default_angle() #+ jp.random_uniform(rng1, (self.sys.num_joint_dof,), -.005, .005)
    qvel = jp.zeros((self.sys.num_joint_dof,)) #jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.005, .005)
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    info = self.sys.info(qp)
    obs = self._get_obs(qp)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_forward': zero,
        'reward_ctrl': zero,
        'reward_healthy': zero,
    }
    return brax_env.State(qp, obs, reward, done, metrics)

  