"""Trains a 2D walker to run in the +x direction."""

from typing import Optional, Tuple

import brax
from brax import jumpy as jp
from brax.envs import env as brax_env
from brax.physics import bodies
from brax.envs import walker2d

class DetWalker2d(walker2d.Walker2d):
  """Trains a 2D walker to run in the +x direction.
  This is similar to the Walker2d-V3 Mujoco environment in OpenAI Gym, which is
  a variant of Hopper with two legs. The two legs do not collide with each
  other.
  """

  def reset(self, rng: jp.ndarray) -> brax_env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jp.random_split(rng, 3)
    qpos = self.sys.default_angle() #+ jp.random_uniform(rng1, (self.sys.num_joint_dof,), -.005, .005)
    qvel = jp.zeros((self.sys.num_joint_dof,)) #jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.005, .005)
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    obs = self._get_obs(qp)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_forward': zero,
        'reward_ctrl': zero,
        'reward_healthy': zero,
    }
    return brax_env.State(qp, obs, reward, done, metrics)
