import jax.numpy as jnp
import brax
from brax import jumpy as jp
from brax.envs import env
from qdax.envs.deterministic_halfcheetah import DetHalfcheetah

class QDUniHalfcheetah(DetHalfcheetah):
    def reset(self, rng: jp.ndarray) -> env.State:
        state = super().reset(rng)
        state.info['contact_cum'] = jp.zeros((8))
        state.info['bd'] = jp.zeros((2))
        return state

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info)
        
        contact = self._get_contact(info)
        contact_cum = state.info['contact_cum'] + contact
        bd = contact_cum/state.info['steps']
        state.info["contact_cum"] = contact_cum
        state.info["bd"] = jnp.array([bd[3], bd[6]]) # BD is only for the contact of the foot and the floor

        x_before = state.qp.pos[0, 0]
        x_after = qp.pos[0, 0]
        forward_reward = (x_after - x_before) / self.sys.config.dt
        ctrl_cost = -.1 * jp.sum(jp.square(action))
        reward = forward_reward + ctrl_cost
        state.metrics.update(
            reward_ctrl_cost=ctrl_cost, reward_forward=forward_reward)

        return state.replace(qp=qp, obs=obs, reward=reward)

    def _get_contact(self, info):
        contact = jp.sum(jp.square(jp.clip(info.contact.vel, -1, 1)), axis=1) != 0
        return contact