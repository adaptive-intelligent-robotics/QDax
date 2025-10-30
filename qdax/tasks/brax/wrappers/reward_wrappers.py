from typing import Optional

import jax
import jax.numpy as jnp
from brax.envs.base import Env, State, Wrapper


class ClipRewardWrapper(Wrapper):
    """Wraps gym environments to clip the reward to be greater than 0.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply clip the reward to be greater than 0.
    """

    def __init__(
        self,
        env: Env,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ) -> None:
        super().__init__(env)
        self._clip_min = clip_min
        self._clip_max = clip_max

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        return state.replace(
            reward=jnp.clip(state.reward, min=self._clip_min, max=self._clip_max)
        )

    def step(self, state: State, action: jax.Array) -> State:
        state = self.env.step(state, action)
        return state.replace(
            reward=jnp.clip(state.reward, min=self._clip_min, max=self._clip_max)
        )


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

    def step(self, state: State, action: jax.Array) -> State:
        state = self.env.step(state, action)
        new_reward = state.reward + self._offset
        return state.replace(reward=new_reward)
