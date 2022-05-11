from abc import abstractmethod
from typing import Any, List, Tuple

from brax import jumpy as jp
from brax.envs.env import Env, State


class QDEnv(Env):
    """
    Wrapper for all QD environments.
    """

    @property
    @abstractmethod
    def state_descriptor_length(self) -> int:
        pass

    @property
    @abstractmethod
    def state_descriptor_name(self) -> str:
        pass

    @property
    @abstractmethod
    def state_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        pass

    @property
    @abstractmethod
    def behavior_descriptor_length(self) -> int:
        pass

    @property
    @abstractmethod
    def behavior_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class QDWrapper(QDEnv):
    """Wrapper for QD environments."""

    def __init__(self, env: QDEnv):
        super().__init__(config=None)
        self.env = env

    def reset(self, rng: jp.ndarray) -> State:
        return self.env.reset(rng)

    def step(self, state: State, action: jp.ndarray) -> State:
        return self.env.step(state, action)

    @property
    def observation_size(self) -> int:
        return self.env.observation_size  # type: ignore

    @property
    def action_size(self) -> int:
        return self.env.action_size  # type: ignore

    @property
    def state_descriptor_length(self) -> int:
        return self.env.state_descriptor_length

    @property
    def state_descriptor_name(self) -> str:
        return self.env.state_descriptor_name

    @property
    def state_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return self.env.state_descriptor_limits

    @property
    def behavior_descriptor_length(self) -> int:
        return self.env.behavior_descriptor_length

    @property
    def behavior_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return self.env.behavior_descriptor_limits

    @property
    def name(self) -> str:
        return self.env.name

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)


class StateDescriptorResetWrapper(QDWrapper):
    """Automatically resets state descriptors."""

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        state.info["first_state_descriptor"] = state.info["state_descriptor"]
        return state

    def step(self, state: State, action: jp.ndarray) -> State:

        state = self.env.step(state, action)

        def where_done(x: jp.ndarray, y: jp.ndarray) -> jp.ndarray:
            done = state.done
            if done.shape:
                done = jp.reshape(done, tuple([x.shape[0]] + [1] * (len(x.shape) - 1)))
            return jp.where(done, x, y)

        state.info["state_descriptor"] = where_done(
            state.info["first_state_descriptor"], state.info["state_descriptor"]
        )
        return state
