"""Trains a halfcheetah to run in the +x direction."""

import jax.numpy as jnp
import brax
from brax import jumpy as jp
from brax.envs import env
from brax.envs import halfcheetah

class DetHalfcheetah(halfcheetah.Halfcheetah):
  """Trains a halfcheetah to run in the +x direction. make environment reset deterministic"""

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jp.random_split(rng, 3)
    qpos = self.sys.default_angle() #+ jp.random_uniform(rng1, (self.sys.num_joint_dof,), -.1, .1)
    qvel = jp.zeros((self.sys.num_joint_dof,)) #jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.1, .1)
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_ctrl_cost': zero,
        'reward_forward': zero,
    }
    return env.State(qp, obs, reward, done, metrics)
