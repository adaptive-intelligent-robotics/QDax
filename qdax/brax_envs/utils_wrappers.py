from brax import jumpy as jp
from brax.envs import env


class StateDescriptorResetWrapper(env.Wrapper):
    """Automatically resets state descriptors."""

    def reset(self, rng: jp.ndarray) -> env.State:
        state = self.env.reset(rng)
        state.info["first_state_descriptor"] = state.info["state_descriptor"]
        return state

    def step(self, state: env.State, action: jp.ndarray) -> env.State:

        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(done, x, y)

        state.info["state_descriptor"] = where_done(
            state.info["first_state_descriptor"], state.info["state_descriptor"]
        )
        return state
