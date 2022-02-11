import brax
from brax import jumpy as jp
from brax.envs import env
from brax.physics import bodies
from brax.envs import humanoid

class DetHumanoid(humanoid.Humanoid):
  """Fitness is velocity in the +x direction """

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jp.random_split(rng, 3)
    qpos = self.sys.default_angle() #+ jp.random_uniform(rng1, (self.sys.num_joint_dof,), -.01, .01)
    qvel = jp.zeros((self.sys.num_joint_dof,)) #jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.01, .01)
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info, jp.zeros(self.action_size))
    self.done = False
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_linvel': zero,
        'reward_quadctrl': zero,
        'reward_alive': zero,
        'reward_impact': zero
    }
    return env.State(qp, obs, reward, done, metrics)
