import brax
from brax import jumpy as jp
from brax.envs import env
from qdax.envs.deterministic_humanoid import DetHumanoid

class QDOmniHumanoid(DetHumanoid):
    def reset(self, rng: jp.ndarray) -> env.State:
        state = super().reset(rng)
        state.info['bd'] = jp.zeros((2))
        return state

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info, action)

        state.info['bd'] = state.qp.pos[0,0:2] # BD is the x-y position which is the first two dimensions of pos

        pos_before = state.qp.pos[:-1]  # ignore floor at last index
        pos_after = qp.pos[:-1]  # ignore floor at last index
        com_before = jp.sum(pos_before * self.mass, axis=0) / jp.sum(self.mass)
        com_after = jp.sum(pos_after * self.mass, axis=0) / jp.sum(self.mass)
        lin_vel_cost = 1.25 * (com_after[0] - com_before[0]) / self.sys.config.dt
        quad_ctrl_cost = .01 * jp.sum(jp.square(action))
        # can ignore contact cost, see: https://github.com/openai/gym/issues/1541
        quad_impact_cost = jp.float32(0)
        alive_bonus = jp.float32(5)


        done = jp.where(qp.pos[0, 2] < 0.65, jp.float32(1), jp.float32(0))
        done = jp.where(qp.pos[0, 2] > 2.1, jp.float32(1), done)

        self.done = jp.where((done == 1), True, self.done)
        reward = jp.where(self.done, jp.float32(-5),lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus)

        state.metrics.update(
            reward_linvel=lin_vel_cost,
            reward_quadctrl=quad_ctrl_cost,
            reward_alive=alive_bonus,
            reward_impact=quad_impact_cost)

        return state.replace(qp=qp, obs=obs, reward=reward, done=done)
