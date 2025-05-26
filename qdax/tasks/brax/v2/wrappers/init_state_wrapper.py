from typing import Callable, Optional

import jax.numpy as jnp
from brax.envs.base import Env, State, Wrapper


class FixedInitialStateWrapper(Wrapper):
    """Wrapper to make the initial state of the environment deterministic and fixed.
    This is done by removing the random noise from the DoF positions and velocities.
    """

    def __init__(
        self,
        env: Env,
        base_env_name: str,
        get_obs_fn: Optional[Callable[[State, jnp.ndarray], jnp.ndarray]] = None,
    ):
        env_get_obs = {
            "hopper": lambda pipeline_state, action: self._get_obs(pipeline_state),
            "walker2d": lambda pipeline_state, action: self._get_obs(pipeline_state),
            "halfcheetah": lambda pipeline_state, action: self._get_obs(pipeline_state),
            "ant": lambda pipeline_state, action: self._get_obs(pipeline_state),
            "humanoid": lambda pipeline_state, action: self._get_obs(
                pipeline_state, action
            ),
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

    def reset(self, rng: jnp.ndarray) -> State:
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

        # Compute new initial positions and velocities
        q = self.env.sys.init_q
        qd = jnp.zeros((self.env.sys.qd_size(),))
        pipeline_state = self.env.pipeline_init(q, qd)

        # get the new obs
        obs = self._get_obs_fn(pipeline_state, jnp.zeros(self.action_size))

        return state.replace(pipeline_state=pipeline_state, obs=obs)
