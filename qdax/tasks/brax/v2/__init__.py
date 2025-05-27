import functools
from typing import Any, Callable, List, Optional, Union

import brax
from brax.envs.base import Env
from brax.envs.wrappers import training

from qdax.tasks.brax.descriptor_extractors import (
    get_feet_contact_proportion,
    get_final_xy_position,
)
from qdax.tasks.brax.v2.envs.base_env import QDEnv
from qdax.tasks.brax.v2.wrappers.base_wrappers import StateDescriptorResetWrapper
from qdax.tasks.brax.v2.wrappers.eval_metrics_wrapper import CompletedEvalWrapper
from qdax.tasks.brax.v2.wrappers.init_state_wrapper import FixedInitialStateWrapper
from qdax.tasks.brax.v2.wrappers.locomotion_wrappers import (
    FeetContactWrapper,
    NoForwardRewardWrapper,
    XYPositionWrapper,
)

# experimentally determined offset (except for antmaze)
# should be sufficient to have only positive rewards but no guarantee
reward_offset = {
    "ant_omni": 3.0,
    "humanoid_omni": 0.0,
    "ant_uni": 3.24,
    "humanoid_uni": 0.0,
    "halfcheetah_uni": 9.231,
    "hopper_uni": 0.9,
    "walker2d_uni": 1.413,
}

descriptor_extractor = {
    "ant_omni": get_final_xy_position,
    "humanoid_omni": get_final_xy_position,
    "ant_uni": get_feet_contact_proportion,
    "humanoid_uni": get_feet_contact_proportion,
    "halfcheetah_uni": get_feet_contact_proportion,
    "hopper_uni": get_feet_contact_proportion,
    "walker2d_uni": get_feet_contact_proportion,
}

_qdax_custom_envs = {
    # All omni envs require debug=False as we don't need to check contacts
    # with the ground
    "ant_omni": {
        "env": "ant",
        "wrappers": [XYPositionWrapper, NoForwardRewardWrapper],
        "kwargs": [{"minval": [-30.0, -30.0], "maxval": [30.0, 30.0]}, {}],
        "debug": False,
    },
    "humanoid_omni": {
        "env": "humanoid",
        "wrappers": [XYPositionWrapper, NoForwardRewardWrapper],
        "kwargs": [{"minval": [-30.0, -30.0], "maxval": [30.0, 30.0]}, {}],
        "debug": False,
    },
    # All uni envs require debug=True to check contacts with the ground
    "ant_uni": {
        "env": "ant",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}],
        "debug": True,
    },
    "humanoid_uni": {
        "env": "humanoid",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}],
        "debug": True,
    },
    "halfcheetah_uni": {
        "env": "halfcheetah",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}],
        "debug": True,
    },
    "hopper_uni": {
        "env": "hopper",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}],
        "debug": True,
    },
    "walker2d_uni": {
        "env": "walker2d",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}],
        "debug": True,
    },
}


def create(
    env_name: str,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    eval_metrics: bool = False,
    fixed_init_state: bool = False,
    qdax_wrappers_kwargs: Optional[List] = None,
    backend: str = "spring",
    **kwargs: Any,
) -> Union[Env, QDEnv]:
    """Creates an Env with a specified brax system.
    Please use namespace to avoid confusion between this function and
    brax.envs.create.
    """

    if env_name in brax.envs._envs.keys():
        env = brax.envs._envs[env_name](backend=backend, **kwargs)
    elif env_name in _qdax_custom_envs.keys():
        base_env_name = _qdax_custom_envs[env_name]["env"]
        is_debug = _qdax_custom_envs[env_name]["debug"]
        if base_env_name in brax.envs._envs.keys():
            env = brax.envs._envs[base_env_name](
                backend=backend, debug=is_debug, **kwargs
            )
        else:
            raise NotImplementedError("This environment name does not exist!")
    else:
        raise NotImplementedError("This environment name does not exist!")

    if env_name in _qdax_custom_envs.keys():
        # roll with qdax wrappers
        wrappers = _qdax_custom_envs[env_name]["wrappers"]
        if qdax_wrappers_kwargs is None:
            kwargs_list = _qdax_custom_envs[env_name]["kwargs"]
        else:
            kwargs_list = qdax_wrappers_kwargs
        for wrapper, kwargs in zip(wrappers, kwargs_list):  # type: ignore
            env = wrapper(env, base_env_name, **kwargs)  # type: ignore

    if episode_length is not None:
        env = training.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = training.VmapWrapper(env, batch_size)
    if fixed_init_state:
        # retrieve the base env
        if env_name not in _qdax_custom_envs.keys():
            base_env_name = env_name
        # wrap the env
        env = FixedInitialStateWrapper(env, base_env_name=base_env_name)  # type: ignore
    if auto_reset:
        env = training.AutoResetWrapper(env)
        if env_name in _qdax_custom_envs.keys():
            env = StateDescriptorResetWrapper(env)

    if eval_metrics:
        env = training.EvalWrapper(env)
        env = CompletedEvalWrapper(env)
    return env


def create_fn(env_name: str, **kwargs: Any) -> Callable[..., Env]:
    """Returns a function that when called, creates an Env.
    Please use namespace to avoid confusion between this function and
    brax.envs.create_fn.
    """
    return functools.partial(create, env_name, **kwargs)
