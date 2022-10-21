import warnings

import brax
import jax.numpy as jnp
from brax import jumpy as jp
from brax.envs import State, env
from google.protobuf import text_format  # type: ignore

from qdax.environments.locomotion_wrappers import COG_NAMES

# config of the body of the trap following Brax's config style
TRAP_CONFIG = """bodies {
    name: "Trap"
    colliders {
        position { x: 12.0 y: 0.0 z: 0.5 }
        rotation { x: 90 y: 0 }
        capsule {
            radius: 0.5
            length: 5
            end: 0
        }
    }
    colliders {
        position { x: 9.5 y: 3.0 z: 0.5 }
        rotation { x: 0 y: 90 }
        capsule {
            radius: 0.5
            length: 6
            end: 0
        }
    }
    colliders {
        position { x: 9.5 y: -3.0 z: 0.5 }
        rotation { x: 0 y: 90 }
        capsule {
            radius: 0.5
            length: 6
            end: 0
        }
    }
    inertia { x: 10000.0 y: 10000.0 z: 10000.0 }
    mass: 1
    frozen { all: true }
}
"""
# config describing collisions of the trap following Brax's config style
# specific to the ant env
ANT_TRAP_COLLISIONS = """collide_include {
    first: "$ Torso"
    second: "Trap"
}
collide_include {
    first: "$ Body 4"
    second: "Trap"
}
collide_include {
    first: "$ Body 7"
    second: "Trap"
}
collide_include {
    first: "$ Body 10"
    second: "Trap"
}
collide_include {
    first: "$ Body 13"
    second: "Trap"
}
collide_include {
    first: "Trap"
    second: "Ground"
}
"""

HUMANOID_TRAP_COLLISIONS = """collide_include {
    first: "left_shin"
    second: "Trap"
}
collide_include {
    first: "right_shin"
    second: "Trap"
}
collide_include {
    first: "Trap"
    second: "Ground"
}
"""

# storing the classic env configurations
# those are the configs from the official brax repo
ENV_SYSTEM_CONFIG = {
    "ant": brax.envs.ant._SYSTEM_CONFIG,
    "halfcheetah": brax.envs.half_cheetah._SYSTEM_CONFIG,
    "walker2d": brax.envs.walker2d._SYSTEM_CONFIG,
    "hopper": brax.envs.hopper._SYSTEM_CONFIG,
    # "humanoid": brax.envs.humanoid._SYSTEM_CONFIG,
}

# linking each env with its specific collision description
# could made more automatic in the future
ENV_TRAP_COLLISION = {
    "ant": ANT_TRAP_COLLISIONS,
    "humanoid": HUMANOID_TRAP_COLLISIONS,
}


class TrapWrapper(env.Wrapper):
    """Wraps gym environments to add a Trap in the environment.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply add the Trap to the environment.

    This wrapper also adds xy in the observation, as it is an important
    information for an agent. Now that there is a trap in its env, we
    expect its actions to depend on its xy position.

    The xy position is normalised thanks to the decided limits of the env,
    which are [0, 30] for x and [-8, 8] for y.

    The only supported envs at the moment are among the classic
    locomotion envs : Ant.

    RMQ: Humanoid is not supported yet.
    RMQ: works for walker2d etc.. but it does not make sens as they
    can only go in one direction.


    Example :

        from brax import envs
        from brax import jumpy as jp

        # choose in ["ant"]
        ENV_NAME = "ant"
        env = envs.create(env_name=ENV_NAME)
        qd_env = TrapWrapper(env, ENV_NAME)

        state = qd_env.reset(rng=jp.random_prngkey(seed=0))
        for i in range(10):
            action = jp.zeros((qd_env.action_size,))
            state = qd_env.step(state, action)


    """

    def __init__(self, env: env.Env, env_name: str) -> None:
        if (
            env_name not in ENV_SYSTEM_CONFIG.keys()
            or env_name not in COG_NAMES.keys()
            or env_name not in ENV_TRAP_COLLISION.keys()
        ):
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")
        if env_name not in ["ant", "humanoid"]:
            warnings.warn("Make sure your agent can move in two dimensions!")
        super().__init__(env)
        self._env_name = env_name
        # update the env config to add the trap
        self._config = (
            ENV_SYSTEM_CONFIG[env_name] + TRAP_CONFIG + ENV_TRAP_COLLISION[env_name]
        )
        # update the associated physical system
        config = text_format.Parse(self._config, brax.Config())
        if not hasattr(self.unwrapped, "sys"):
            raise AttributeError("Cannot link env to a physical system.")
        self.unwrapped.sys = brax.System(config)
        self._cog_idx = self.unwrapped.sys.body.index[COG_NAMES[env_name]]

        # we need to normalise x/y position to avoid values to explose
        self._substract = jnp.array([15, 0])  # come from env limits
        self._divide = jnp.array([15, 8])  # come from env limits

    @property
    def name(self) -> str:
        return self._env_name

    @property
    def observation_size(self) -> int:
        """The size of the observation vector returned in step and reset."""
        rng = jp.random_prngkey(0)
        reset_state = self.reset(rng)
        return int(reset_state.obs.shape[-1])

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        # add xy position to the observation
        xy_pos = state.qp.pos[self._cog_idx][:2]
        # normalise
        xy_pos = (xy_pos - self._substract) / self._divide
        new_obs = jp.concatenate([xy_pos, state.obs])
        return state.replace(obs=new_obs)  # type: ignore

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        # add xy position to the observation
        xy_pos = state.qp.pos[self._cog_idx][:2]
        # normalise
        xy_pos = (xy_pos - self._substract) / self._divide
        new_obs = jp.concatenate([xy_pos, state.obs])
        return state.replace(obs=new_obs)  # type: ignore


# maze body part
MAZE_CONFIG = """bodies {
  name: "Maze"
  colliders {
    position { x: 17.5 y: -5.0 z: 0.5 }
    rotation { x: 0 y: 90 }
    capsule {
      radius: 0.5
      length: 46
      end: 0
    }
  }
  colliders {
    position { x: 17.5 y: 40.0 z: 0.5 }
    rotation { x: 0 y: 90 }
    capsule {
      radius: 0.5
      length: 46
      end: 0
    }
  }
  colliders {
    position { x: -5.0 y: 17.5 z: 0.5 }
    rotation { x: 90 y: 0 }
    capsule {
      radius: 0.5
      length: 44
      end: 0
    }
  }
  colliders {
    position { x: 40.0 y: 17.5 z: 0.5 }
    rotation { x: 90 y: 0 }
    capsule {
      radius: 0.5
      length: 44
      end: 0
    }
  }
  colliders {
    position { x: 20.0 y: 7.5 z: 0.5 }
    rotation { x: 90 y: 0 }
    capsule {
      radius: 0.5
      length: 24
      end: 0
    }
  }
  colliders {
    position { x: 15.0 y: 7.5 z: 0.5 }
    rotation { x: 0 y: 90 }
    capsule {
      radius: 0.5
      length: 9.0
      end: 0
    }
  }
  colliders {
    position { x: 10.0 y: 30.0 z: 0.5 }
    rotation { x: 90 y: 0 }
    capsule {
      radius: 0.5
      length: 19.5
      end: 0
    }
  }
  inertia { x: 10000.0 y: 10000.0 z: 10000.0 }
  mass: 1
  frozen { all: true }
}
bodies {
    name: "Target"
    colliders {
        sphere { radius: 0.5 }
    }
    frozen { all: true }
}
defaults {
  qps {
    name: "Target"
    pos { x: 35.0 y: 0.0 z: 0.5 }
  }
}
"""

# describe the physical collisions
ANT_MAZE_COLLISIONS = """collide_include {
    first: "Maze"
    second: "Ground"
}
collide_include {
    first: "$ Torso"
    second: "Maze"
}
collide_include {
    first: "$ Body 4"
    second: "Maze"
}
collide_include {
    first: "$ Body 7"
    second: "Maze"
}
collide_include {
    first: "$ Body 10"
    second: "Maze"
}
collide_include {
    first: "$ Body 13"
    second: "Maze"
}
collide_include {
    first: "Maze"
    second: "Ground"
}
"""

ENV_MAZE_COLLISION = {
    "ant": ANT_MAZE_COLLISIONS,
}


class MazeWrapper(env.Wrapper):
    """Wraps gym environments to add a maze in the environment
    and a new reward (distance to the goal).

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply add the Maze to the environment,
    along with the new reward.

    This wrapper also adds xy in the observation, as it is an important
    information for an agent. Now that the agent is in a maze, we
    expect its actions to depend on its xy position.

    The xy position is normalised thanks to the decided limits of the env,
    which are [-5, 40] for x and y.

    The only supported envs at the moment are among the classic
    locomotion envs : Ant.

    RMQ: Humanoid is not supported yet.
    RMQ: works for walker2d etc.. but it does not make sens as they
    can only go in one direction.

    Example :

        from brax import envs
        from brax import jumpy as jp

        # choose in ["ant"]
        ENV_NAME = "ant"
        env = envs.create(env_name=ENV_NAME)
        qd_env = MazeWrapper(env, ENV_NAME)

        state = qd_env.reset(rng=jp.random_prngkey(seed=0))
        for i in range(10):
            action = jp.zeros((qd_env.action_size,))
            state = qd_env.step(state, action)


    """

    def __init__(self, env: env.Env, env_name: str) -> None:
        if (
            env_name not in ENV_SYSTEM_CONFIG.keys()
            or env_name not in COG_NAMES.keys()
            or env_name not in ENV_MAZE_COLLISION.keys()
        ):
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")
        if env_name not in ["ant", "humanoid"]:
            warnings.warn("Make sure your agent can move in two dimensions!")
        super().__init__(env)
        self._env_name = env_name
        self._config = (
            ENV_SYSTEM_CONFIG[env_name] + MAZE_CONFIG + ENV_MAZE_COLLISION[env_name]
        )
        config = text_format.Parse(self._config, brax.Config())
        if not hasattr(self.unwrapped, "sys"):
            raise AttributeError("Cannot link env to a physical system.")
        self.unwrapped.sys = brax.System(config)
        self._cog_idx = self.unwrapped.sys.body.index[COG_NAMES[env_name]]
        self._target_idx = self.unwrapped.sys.body.index["Target"]

        # we need to normalise x/y position to avoid values to explose
        self._substract = jnp.array([17.5, 17.5])  # come from env limits
        self._divide = jnp.array([22.5, 22.5])  # come from env limits

    @property
    def name(self) -> str:
        return self._env_name

    @property
    def observation_size(self) -> int:
        """The size of the observation vector returned in step and reset."""
        rng = jp.random_prngkey(0)
        reset_state = self.reset(rng)
        return int(reset_state.obs.shape[-1])

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        # get xy position of the center of gravity and of the target
        cog_xy_position = state.qp.pos[self._cog_idx][:2]
        target_xy_position = state.qp.pos[self._target_idx][:2]
        # update the reward
        new_reward = -jp.norm(target_xy_position - cog_xy_position)
        # add cog xy position to the observation - normalise
        cog_xy_position = (cog_xy_position - self._substract) / self._divide
        new_obs = jp.concatenate([cog_xy_position, state.obs])
        return state.replace(obs=new_obs, reward=new_reward)  # type: ignore

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        # get xy position of the center of gravity and of the target
        cog_xy_position = state.qp.pos[self._cog_idx][:2]
        target_xy_position = state.qp.pos[self._target_idx][:2]
        # update the reward
        new_reward = -jp.norm(target_xy_position - cog_xy_position)
        # add cog xy position to the observation - normalise
        cog_xy_position = (cog_xy_position - self._substract) / self._divide
        new_obs = jp.concatenate([cog_xy_position, state.obs])
        # brax ant suicides by jumping over a manually designed z threshold
        # this line avoid this by increasing the threshold
        done = jp.where(
            state.qp.pos[0, 2] < 0.2,
            x=jp.array(1, dtype=jp.float32),
            y=jp.array(0, dtype=jp.float32),
        )
        done = jp.where(
            state.qp.pos[0, 2] > 5.0, x=jp.array(1, dtype=jp.float32), y=done
        )
        return state.replace(obs=new_obs, reward=new_reward, done=done)  # type: ignore
