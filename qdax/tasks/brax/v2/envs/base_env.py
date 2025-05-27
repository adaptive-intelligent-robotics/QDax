from abc import abstractmethod
from typing import Tuple

import jax
from brax.envs.base import Env


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
    def state_descriptor_limits(self) -> Tuple[jax.Array, jax.Array]:
        pass

    @property
    @abstractmethod
    def descriptor_length(self) -> int:
        pass

    @property
    @abstractmethod
    def descriptor_limits(self) -> Tuple[jax.Array, jax.Array]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
