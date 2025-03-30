import jax.numpy as jnp
from brax.envs.base import Env, State, Wrapper


class ClipRewardWrapper(Wrapper):
    """Wraps gym environments to clip the reward to be greater than 0.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply clip the reward to be greater than 0.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        return state.replace(reward=jnp.clip(state.reward, a_min=0.0))

    def step(self, state: State, action: jnp.ndarray) -> State:
        state = self.env.step(state, action)
        return state.replace(reward=jnp.clip(state.reward, a_min=0.0))


class OffsetRewardWrapper(Wrapper):
    """Wraps gym environments to offset the reward to be greater than 0.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply clip the reward to be greater than 0.
    """

    def __init__(self, env: Env, offset: float = 0.0) -> None:
        super().__init__(env)
        self._offset = offset

    @property
    def name(self) -> str:
        return str(self._env_name)

    def step(self, state: State, action: jnp.ndarray) -> State:
        state = self.env.step(state, action)
        new_reward = state.reward + self._offset
        return state.replace(reward=new_reward)
