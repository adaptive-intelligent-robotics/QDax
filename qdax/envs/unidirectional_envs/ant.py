import brax
from brax import jumpy as jp
from brax.envs import env
from qdax.envs.deterministic_ant import DetAnt

class QDUniAnt(DetAnt):
    def reset(self, rng: jp.ndarray) -> env.State:
        state = super().reset(rng)
        state.info['contact_cum'] = jp.zeros((10))
        state.info['bd'] = jp.zeros((4))
        return state

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info)

        contact = self._get_contact(info)
        contact_cum = state.info['contact_cum'] + contact
        bd = contact_cum/state.info['steps']
        state.info["contact_cum"] = contact_cum
        state.info["bd"] = bd[2:-1:2] # BD is only for the contact of the foot and the floor

        x_before = state.qp.pos[0, 0]
        x_after = qp.pos[0, 0]
        forward_reward = (x_after - x_before) / self.sys.config.dt
        ctrl_cost = .5 * jp.sum(jp.square(action))
        contact_cost = (0.5 * 1e-3 *
                        jp.sum(jp.square(jp.clip(info.contact.vel, -1, 1))))
        survive_reward = jp.float32(1)
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        done = jp.where(qp.pos[0, 2] < 0.2, x=jp.float32(1), y=jp.float32(0))
        done = jp.where(qp.pos[0, 2] > 1.0, x=jp.float32(1), y=done)
        state.metrics.update(
            reward_ctrl_cost=ctrl_cost,
            reward_contact_cost=contact_cost,
            reward_forward=forward_reward,
            reward_survive=survive_reward)

        return state.replace(qp=qp, obs=obs, reward=reward, done=done)

    def _get_contact(self, info):
        contact = jp.sum(jp.square(jp.clip(info.contact.vel, -1, 1)), axis=1) != 0
        return contact