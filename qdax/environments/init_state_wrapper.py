from brax import jumpy as jp
from brax.envs import Env, State, Wrapper

GET_OBS_API = {
    "ant": ["qp", "info"],
    "halfcheetah": ["qp", "info"],
    "walker2d": ["qp"],
    "hopper": ["qp"],
    "humanoid": ["qp", "info", "action"],
}


class FixedInitialStateWrapper(Wrapper):
    """
    Wrapper to make the initial state of the environment deterministic and fixed.
    This is done by removing the random noise from the DoF positions and velocities.
    """

    def __init__(
        self,
        env: Env,
        base_env_name: str,
    ):
        if base_env_name not in GET_OBS_API.keys():
            raise NotImplementedError(
                f"This wrapper does not support {base_env_name} yet."
            )

        self._input_keys = GET_OBS_API[base_env_name]

        super().__init__(env)

    def reset(self, rng: jp.ndarray) -> State:
        # Run the default reset method of parent environment
        state = self.env.reset(rng)

        # Compute new initial positions and velicities
        qpos = self.sys.default_angle()
        qvel = jp.zeros((self.sys.num_joint_dof,))

        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)

        # build the full input dictionary
        obs_full_input = {
            "qp": qp,
            "info": self.sys.info(qp),
            "action": jp.zeros(self.action_size),
        }

        # extract the relevant input for the env at hand
        obs_specific_input = {key: obs_full_input[key] for key in self._input_keys}

        # unpack the input into the _get_obs function
        obs = self._get_obs(**obs_specific_input)

        # retrieve the udpated state
        return state.replace(qp=qp, obs=obs)
