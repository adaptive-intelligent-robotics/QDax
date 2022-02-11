import brax
from brax import jumpy as jp
from brax.envs import env
from qdax.envs.deterministic_hopper import DetHopper

class QDUniHopper(DetHopper):
    def reset(self, rng: jp.ndarray) -> env.State:
        state = super().reset(rng)
        state.info['contact_cum'] = jp.zeros((5))
        state.info['bd'] = jp.zeros((1))
        return state

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        # Reverse torque improves performance over a range of hparams.
        qp, info = self.sys.step(state.qp, -action)
        obs = self._get_obs(qp) 

        contact = self._get_contact(info)
        contact_cum = state.info['contact_cum'] + contact
        bd = contact_cum/state.info['steps']
        state.info["contact_cum"] = contact_cum
        state.info["bd"] = bd[3] # BD is only for the contact of the foot and the floor

        # Ignore the floor at last index.
        pos_before = state.qp.pos[:-1]
        pos_after = qp.pos[:-1]
        com_before = jp.sum(pos_before * self.mass, axis=0) / jp.sum(self.mass)
        com_after = jp.sum(pos_after * self.mass, axis=0) / jp.sum(self.mass)
        x_velocity = (com_after[0] - com_before[0]) / self.sys.config.dt
        forward_reward = self._forward_reward_weight * x_velocity

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(qp.pos[0, 2] < min_z, x=0.0, y=1.0)
        is_healthy = jp.where(qp.pos[0, 2] > max_z, x=0.0, y=is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        rewards = forward_reward + healthy_reward

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        costs = ctrl_cost

        reward = rewards - costs

        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        state.metrics.update(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_healthy=healthy_reward)
        return state.replace(qp=qp, obs=obs, reward=reward, done=done)


    def _get_contact(self, info):
        contact = jp.sum(jp.square(jp.clip(info.contact.vel, -1, 1)), axis=1) != 0
        return contact

