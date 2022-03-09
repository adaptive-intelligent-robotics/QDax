import brax
import jax.numpy as jnp
from brax import jumpy as jp
from brax.envs import env

from qdax.envs.deterministic_humanoid import DetHumanoid


class QDUniHumanoid(DetHumanoid):
    def reset(self, rng: jp.ndarray) -> env.State:
        state = super().reset(rng)
        state.info["contact_cum"] = jp.zeros((12))
        state.info["bd"] = jp.zeros((2))
        return state

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info, action)

        contact = self._get_contact(info)
        contact_cum = state.info["contact_cum"] + contact
        bd = contact_cum / state.info["steps"]
        state.info["contact_cum"] = contact_cum
        state.info["bd"] = jnp.array(
            [bd[4], bd[6]]
        )  # BD is only for the contact of the foot and the floor

        pos_before = state.qp.pos[:-1]  # ignore floor at last index
        pos_after = qp.pos[:-1]  # ignore floor at last index
        com_before = jp.sum(pos_before * self.mass, axis=0) / jp.sum(self.mass)
        com_after = jp.sum(pos_after * self.mass, axis=0) / jp.sum(self.mass)
        lin_vel_cost = 1.25 * (com_after[0] - com_before[0]) / self.sys.config.dt
        quad_ctrl_cost = 0.01 * jp.sum(jp.square(action))
        # can ignore contact cost, see: https://github.com/openai/gym/issues/1541
        quad_impact_cost = jp.float32(0)
        alive_bonus = jp.float32(1)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus

        done = jp.where(qp.pos[0, 2] < 0.7, jp.float32(1), jp.float32(0))
        done = jp.where(qp.pos[0, 2] > 2.1, jp.float32(1), done)
        state.metrics.update(
            reward_linvel=lin_vel_cost,
            reward_quadctrl=quad_ctrl_cost,
            reward_alive=alive_bonus,
            reward_impact=quad_impact_cost,
        )

        return state.replace(qp=qp, obs=obs, reward=reward, done=done)

    def _get_contact(self, info):
        contact = jp.sum(jp.square(jp.clip(info.contact.vel, -1, 1)), axis=1) != 0
        return contact
