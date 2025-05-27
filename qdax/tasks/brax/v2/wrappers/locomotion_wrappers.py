from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
from brax.envs.base import Env, State, Wrapper

from qdax.tasks.brax.v2.wrappers.base_wrappers import QDWrapper

FEET_NAMES = {
    "hopper": ["foot"],
    "walker2d": ["foot", "foot_left"],
    "halfcheetah": ["bfoot", "ffoot"],
    "ant": ["$ Body 4", "$ Body 7", "$ Body 10", "$ Body 13"],
    "humanoid": ["left_shin", "right_shin"],
}


class FeetContactWrapper(QDWrapper):
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
        import jax.numpy as jnp

        # choose in ["ant", "walker2d", "hopper", "halfcheetah"]
        ENV_NAME = "ant"
        env = envs.create(env_name=ENV_NAME)
        qd_env = FeetContactWrapper(env, ENV_NAME)

        state = qd_env.reset(rng=jax.random.PRNGKey(0))
        for i in range(10):
            action = jnp.zeros((qd_env.action_size,))
            state = qd_env.step(state, action)

            # retrieve feet contact
            feet_contact = state.info["state_descriptor"]

            # do whatever you want with feet_contact
            print(f"Feet contact : {feet_contact}")


    """

    def __init__(self, env: Env, env_name: str):
        if env_name not in FEET_NAMES.keys():
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        super().__init__(env)
        self.env = env
        self._env_name = env_name

        if hasattr(self.env, "sys"):
            self._feet_idx = jnp.array(
                [
                    i
                    for i, feet_name in enumerate(self.env.sys.link_names)
                    if feet_name in FEET_NAMES[env_name]
                ]
            )
        else:
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

    @property
    def state_descriptor_length(self) -> int:
        return self.descriptor_length

    @property
    def state_descriptor_name(self) -> str:
        return "feet_contact"

    @property
    def state_descriptor_limits(self) -> Tuple[List, List]:
        return self.descriptor_limits

    @property
    def descriptor_length(self) -> int:
        return len(self._feet_idx)

    @property
    def descriptor_limits(self) -> Tuple[List, List]:
        bd_length = self.descriptor_length
        return (jnp.zeros((bd_length,)), jnp.ones((bd_length,)))

    @property
    def name(self) -> str:
        return self._env_name

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        state.info["state_descriptor"] = self._get_feet_contact(state)
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        state = self.env.step(state, action)
        state.info["state_descriptor"] = self._get_feet_contact(state)
        return state

    def _get_feet_contact(self, state: State) -> jnp.ndarray:
        return jnp.any(
            jax.vmap(
                lambda x: (state.pipeline_state.contact.link_idx[1] == x)
                & (state.pipeline_state.contact.dist <= 0)
            )(self._feet_idx),
            axis=-1,
        ).astype(jnp.float32)

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)


# name of the center of gravity
COG_NAMES = {
    "hopper": "torso",
    "walker2d": "torso",
    "halfcheetah": "torso",
    "ant": "torso",
    "humanoid": "torso",
}


class XYPositionWrapper(QDWrapper):
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
        import jax.numpy as jnp

        # choose in ["ant", "walker2d", "hopper", "halfcheetah", "humanoid"]
        ENV_NAME = "ant"
        env = envs.create(env_name=ENV_NAME)
        qd_env = XYPositionWrapper(env, ENV_NAME)

        state = qd_env.reset(rng=jax.random.PRNGKey(0))
        for i in range(10):
            action = jnp.zeros((qd_env.action_size,))
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

        super().__init__(env)
        self.env = env
        self._env_name = env_name

        if hasattr(self.env, "sys"):
            self._cog_idx = self.env.sys.link_names.index(COG_NAMES[env_name])
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
    def descriptor_length(self) -> int:
        return self.state_descriptor_length

    @property
    def descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return self.state_descriptor_limits

    @property
    def name(self) -> str:
        return self._env_name

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        state.info["state_descriptor"] = jnp.clip(
            state.pipeline_state.x.pos[self._cog_idx][:2],
            a_min=self._minval,
            a_max=self._maxval,
        )
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        state = self.env.step(state, action)
        state.info["state_descriptor"] = jnp.clip(
            state.pipeline_state.x.pos[self._cog_idx][:2],
            a_min=self._minval,
            a_max=self._maxval,
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
    "halfcheetah": "reward_run",
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
        import jax.numpy as jnp

        # choose in ["ant", "walker2d", "hopper", "halfcheetah", "humanoid"]
        ENV_NAME = "ant"
        env = envs.create(env_name=ENV_NAME)
        qd_env = NoForwardRewardWrapper(env, ENV_NAME)

        state = qd_env.reset(rng=jax.random.PRNGKey(0))
        for i in range(10):
            action = jnp.zeros((qd_env.action_size,))
            state = qd_env.step(state, action)
    """

    def __init__(self, env: Env, env_name: str) -> None:
        if env_name not in FORWARD_REWARD_NAMES.keys():
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        super().__init__(env)
        self._env_name = env_name
        self._forward_reward_name = FORWARD_REWARD_NAMES[env_name]

    @property
    def name(self) -> str:
        return self._env_name

    def step(self, state: State, action: jnp.ndarray) -> State:
        state = self.env.step(state, action)
        new_reward = state.reward - state.metrics[self._forward_reward_name]
        return state.replace(reward=new_reward)  # type: ignore
