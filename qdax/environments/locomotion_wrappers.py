from typing import Any, List, Optional, Sequence, Tuple

import jax.numpy as jnp
from brax import jumpy as jp
from brax.envs import Env, State, Wrapper
from brax.physics import config_pb2
from brax.physics.base import QP, Info
from brax.physics.system import System

from qdax.environments.base_wrappers import QDEnv

FEET_NAMES = {
    "ant": ["$ Body 4", "$ Body 7", "$ Body 10", "$ Body 13"],
    "halfcheetah": ["ffoot", "bfoot"],
    "walker2d": ["foot", "foot_left"],
    "hopper": ["foot"],
    "humanoid": ["left_shin", "right_shin"],
}


class QDSystem(System):
    """Inheritance of brax physic system.

    Work precisely the same but store some information from the physical
    simulation in the aux_info attribute.

    This is used in FeetContactWrapper to get the feet contact of the
    robot with the ground.
    """

    def __init__(
        self, config: config_pb2.Config, resource_paths: Optional[Sequence[str]] = None
    ):
        super().__init__(config, resource_paths=resource_paths)
        self.aux_info = None

    def step(self, qp: QP, act: jp.ndarray) -> Tuple[QP, Info]:
        qp, info = super().step(qp, act)
        self.aux_info = info
        return qp, info


class FeetContactWrapper(QDEnv):
    """Wraps gym environments to add the feet contact data.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply add the feet_contact booleans in
    the information dictionary of the Brax.state.

    The only supported envs at the moment are among the classic
    locomotion envs : Walker2D, Hopper, Ant, Bullet.

    New locomotions envs can easily be added by adding the config name
    of the feet of the corresponding environment in the FEET_NAME dictionary.

    Example :

        from brax import envs
        from brax import jumpy as jp

        # choose in ["ant", "walker2d", "hopper", "halfcheetah"]
        ENV_NAME = "ant"
        env = envs.create(env_name=ENV_NAME)
        qd_env = FeetContactWrapper(env, ENV_NAME)

        state = qd_env.reset(rng=jp.random_prngkey(seed=0))
        for i in range(10):
            action = jp.zeros((qd_env.action_size,))
            state = qd_env.step(state, action)

            # retrieve feet contact
            feet_contact = state.info["state_descriptor"]

            # do whatever you want with feet_contact
            print(f"Feet contact : {feet_contact}")


    """

    def __init__(self, env: Env, env_name: str):

        if env_name not in FEET_NAMES.keys():
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        super().__init__(config=None)

        self.env = env
        self._env_name = env_name
        if hasattr(self.env, "sys"):
            self.env.sys = QDSystem(self.env.sys.config)

        self._feet_contact_idx = jp.array(
            [env.sys.body.index.get(name) for name in FEET_NAMES[env_name]]
        )

    @property
    def state_descriptor_length(self) -> int:
        return self.behavior_descriptor_length

    @property
    def state_descriptor_name(self) -> str:
        return "feet_contact"

    @property
    def state_descriptor_limits(self) -> Tuple[List, List]:
        return self.behavior_descriptor_limits

    @property
    def behavior_descriptor_length(self) -> int:
        return len(self._feet_contact_idx)

    @property
    def behavior_descriptor_limits(self) -> Tuple[List, List]:
        bd_length = self.behavior_descriptor_length
        return (jnp.zeros((bd_length,)), jnp.ones((bd_length,)))

    @property
    def name(self) -> str:
        return self._env_name

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        state.info["state_descriptor"] = self._get_feet_contact(
            self.env.sys.info(state.qp)
        )
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        state.info["state_descriptor"] = self._get_feet_contact(self.env.sys.aux_info)
        return state

    def _get_feet_contact(self, info: Info) -> jp.ndarray:
        contacts = info.contact.vel
        return jp.any(contacts[self._feet_contact_idx], axis=1).astype(jp.float32)

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)


# name of the center of gravity
COG_NAMES = {
    "ant": "$ Torso",
    "halfcheetah": "torso",
    "walker2d": "torso",
    "hopper": "torso",
    "humanoid": "torso",
}


class XYPositionWrapper(QDEnv):
    """Wraps gym environments to add the position data.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply add the actual position in
    the information dictionary of the Brax.state.

    One can also add values to clip the state descriptors.

    The only supported envs at the moment are among the classic
    locomotion envs : Ant, Humanoid.

    New locomotions envs can easily be added by adding the config name
    of the feet of the corresponding environment in the STATE_POSITION
    dictionary.

    RMQ: this can be used with Hopper, Walker2d, Halfcheetah but it makes
    less sens as those are limited to one direction.

    Example :

        from brax import envs
        from brax import jumpy as jp

        # choose in ["ant", "walker2d", "hopper", "halfcheetah", "humanoid"]
        ENV_NAME = "ant"
        env = envs.create(env_name=ENV_NAME)
        qd_env = XYPositionWrapper(env, ENV_NAME)

        state = qd_env.reset(rng=jp.random_prngkey(seed=0))
        for i in range(10):
            action = jp.zeros((qd_env.action_size,))
            state = qd_env.step(state, action)

            # retrieve feet contact
            xy_position = state.info["xy_position"]

            # do whatever you want with xy_position
            print(f"xy position : {xy_position}")


    """

    def __init__(
        self,
        env: Env,
        env_name: str,
        minval: Optional[List[float]] = None,
        maxval: Optional[List[float]] = None,
    ):
        if env_name not in COG_NAMES.keys():
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        super().__init__(config=None)

        self.env = env
        self._env_name = env_name
        if hasattr(self.env, "sys"):
            self._cog_idx = self.env.sys.body.index[COG_NAMES[env_name]]
        else:
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        if minval is None:
            minval = jnp.ones((2,)) * (-jnp.inf)

        if maxval is None:
            maxval = jnp.ones((2,)) * jnp.inf

        if len(minval) == 2 and len(maxval) == 2:
            self._minval = jnp.array(minval)
            self._maxval = jnp.array(maxval)
        else:
            raise NotImplementedError(
                "Please make sure to give two values for each limits."
            )

    @property
    def state_descriptor_length(self) -> int:
        return 2

    @property
    def state_descriptor_name(self) -> str:
        return "xy_position"

    @property
    def state_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return self._minval, self._maxval

    @property
    def behavior_descriptor_length(self) -> int:
        return self.state_descriptor_length

    @property
    def behavior_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return self.state_descriptor_limits

    @property
    def name(self) -> str:
        return self._env_name

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        state.info["state_descriptor"] = jnp.clip(
            state.qp.pos[self._cog_idx][:2], a_min=self._minval, a_max=self._maxval
        )
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        # get xy position of the center of gravity
        state.info["state_descriptor"] = jnp.clip(
            state.qp.pos[self._cog_idx][:2], a_min=self._minval, a_max=self._maxval
        )
        return state

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)


# name of the forward/velocity reward
FORWARD_REWARD_NAMES = {
    "ant": "reward_forward",
    "halfcheetah": "reward_forward",
    "walker2d": "reward_forward",
    "hopper": "reward_forward",
    "humanoid": "reward_linvel",
}


class NoForwardRewardWrapper(Wrapper):
    """Wraps gym environments to remove forward reward.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply remove the forward speed term
    of the reward.

    Example :

        from brax import envs
        from brax import jumpy as jp

        # choose in ["ant", "walker2d", "hopper", "halfcheetah", "humanoid"]
        ENV_NAME = "ant"
        env = envs.create(env_name=ENV_NAME)
        qd_env = NoForwardRewardWrapper(env, ENV_NAME)

        state = qd_env.reset(rng=jp.random_prngkey(seed=0))
        for i in range(10):
            action = jp.zeros((qd_env.action_size,))
            state = qd_env.step(state, action)
    """

    def __init__(self, env: Env, env_name: str) -> None:
        if env_name not in FORWARD_REWARD_NAMES.keys():
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")
        super().__init__(env)
        self._env_name = env_name
        self._fd_reward_field = FORWARD_REWARD_NAMES[env_name]

    @property
    def name(self) -> str:
        return self._env_name

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        # update the reward (remove forward_reward)
        new_reward = state.reward - state.metrics[self._fd_reward_field]
        return state.replace(reward=new_reward)  # type: ignore
