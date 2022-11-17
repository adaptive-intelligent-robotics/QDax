from typing import Callable, Optional

import brax
from brax import jumpy as jp
from brax.envs import Env, State, Wrapper


class FixedInitialStateWrapper(Wrapper):
    """Wrapper to make the initial state of the environment deterministic and fixed.
    This is done by removing the random noise from the DoF positions and velocities.
    """

    def __init__(
        self,
        env: Env,
        base_env_name: str,
        get_obs_fn: Optional[
            Callable[[brax.QP, brax.Info, jp.ndarray], jp.ndarray]
        ] = None,
    ):
        env_get_obs = {
            "ant": lambda qp, info, action: self._get_obs(qp, info),
            "halfcheetah": lambda qp, info, action: self._get_obs(qp, info),
            "walker2d": lambda qp, info, action: self._get_obs(qp),
            "hopper": lambda qp, info, action: self._get_obs(qp),
            "humanoid": lambda qp, info, action: self._get_obs(qp, info, action),
        }

        super().__init__(env)

        if get_obs_fn is not None:
            self._get_obs_fn = get_obs_fn
        elif base_env_name in env_get_obs.keys():
            self._get_obs_fn = env_get_obs[base_env_name]
        else:
            raise NotImplementedError(
                f"This wrapper does not support {base_env_name} yet."
            )

    def reset(self, rng: jp.ndarray) -> State:
        """Reset the state of the environment with a deterministic and fixed
        initial state.

        Args:
            rng: random key to handle stochastic operations. Used by the parent
                init reset function.

        Returns:
            A new state with a fixed observation.
        """
        # Run the default reset method of parent environment
        state = self.env.reset(rng)

        # Compute new initial positions and velicities
        qpos = self.sys.default_angle()
        qvel = jp.zeros((self.sys.num_joint_dof,))

        # update qd
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)

        # get the new obs
        obs = self._get_obs_fn(qp, self.sys.info(qp), jp.zeros(self.action_size))

        return state.replace(qp=qp, obs=obs)
