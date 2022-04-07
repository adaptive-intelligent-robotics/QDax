import functools
from typing import Callable, Optional, Union

import brax

from qdax.brax_envs.bd_extractors import (
    get_feet_contact_proportion,
    get_final_xy_position,
)
from qdax.brax_envs.exploration_wrappers import MazeWrapper, TrapWrapper
from qdax.brax_envs.locomotion_wrappers import (
    FeetContactWrapper,
    NoForwardRewardWrapper,
    XYPositionWrapper,
)
from qdax.brax_envs.pointmaze import PointMaze
from qdax.brax_envs.utils_wrappers import QDEnv, StateDescriptorResetWrapper

# experimentally determinate offset (except for antmaze)
# should be efficient to have only positive rewards but no guarantee
reward_offset = {
    "pointmaze": 2.3431,
    "anttrap": 3.38,
    "antmaze": 40.32,
    "ant_omni": 3.0,
    "humanoid_omni": 0.0,
    "ant_uni": 3.24,
    "humanoid_uni": 0.0,
    "halfcheetah_uni": 9.231,
    "hopper_uni": 0.9,
    "walker2d_uni": 1.413,
}

behavior_descriptor_extractor = {
    "pointmaze": get_final_xy_position,
    "anttrap": get_final_xy_position,
    "antmaze": get_final_xy_position,
    "ant_omni": get_final_xy_position,
    "humanoid_omni": get_final_xy_position,
    "ant_uni": get_feet_contact_proportion,
    "humanoid_uni": get_feet_contact_proportion,
    "halfcheetah_uni": get_feet_contact_proportion,
    "hopper_uni": get_feet_contact_proportion,
    "walker2d_uni": get_feet_contact_proportion,
}

_parl_envs = {
    "pointmaze": PointMaze,
}

_parl_custom_envs = {
    "anttrap": {
        "env": "ant",
        "wrappers": [XYPositionWrapper, TrapWrapper],
        "kwargs": [{"minval": [0.0, -8.0], "maxval": [30.0, 8.0]}, {}],
    },
    "antmaze": {
        "env": "ant",
        "wrappers": [XYPositionWrapper, MazeWrapper],
        "kwargs": [{"minval": [-5.0, -5.0], "maxval": [40.0, 40.0]}, {}],
    },
    "ant_omni": {
        "env": "ant",
        "wrappers": [XYPositionWrapper, NoForwardRewardWrapper],
        "kwargs": [{"minval": [-30.0, -30.0], "maxval": [30.0, 30.0]}, {}],
    },
    "humanoid_omni": {
        "env": "humanoid",
        "wrappers": [XYPositionWrapper, NoForwardRewardWrapper],
        "kwargs": [{"minval": [-30.0, -30.0], "maxval": [30.0, 30.0]}, {}],
    },
    "ant_uni": {"env": "ant", "wrappers": [FeetContactWrapper], "kwargs": [{}, {}]},
    "humanoid_uni": {
        "env": "humanoid",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}, {}],
    },
    "halfcheetah_uni": {
        "env": "halfcheetah",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}, {}],
    },
    "hopper_uni": {
        "env": "hopper",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}, {}],
    },
    "walker2d_uni": {
        "env": "walker2d",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}, {}],
    },
}


def create(
    env_name: str,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    eval_metrics: bool = False,
    **kwargs,
) -> Union[brax.envs.Env, QDEnv]:
    """Creates an Env with a specified brax system.
    Please use namespace to avoid confusion between this function and
    brax.envs.create.
    """

    if env_name in brax.envs._envs.keys():
        env = brax.envs._envs[env_name](**kwargs)
    elif env_name in _parl_envs.keys():
        env = _parl_envs[env_name](**kwargs)
    elif env_name in _parl_custom_envs.keys():
        base_env_name = _parl_custom_envs[env_name]["env"]
        env = brax.envs._envs[base_env_name](**kwargs)

        # roll with parl wrappers
        wrappers = _parl_custom_envs[env_name]["wrappers"]
        kwargs_list = _parl_custom_envs[env_name]["kwargs"]

        for wrapper, kwargs in zip(wrappers, kwargs_list):
            env = wrapper(env, base_env_name, **kwargs)
    else:
        raise NotImplementedError("This environment name does not exist!")

    if episode_length is not None:
        env = brax.envs.wrappers.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = brax.envs.wrappers.VectorWrapper(env, batch_size)
    if auto_reset:
        env = brax.envs.wrappers.AutoResetWrapper(env)
        if env_name in _parl_custom_envs.keys():
            env = StateDescriptorResetWrapper(env)
    if eval_metrics:
        env = brax.envs.wrappers.EvalWrapper(env)
    return env


def create_fn(env_name: str, **kwargs) -> Callable[..., brax.envs.Env]:
    """Returns a function that when called, creates an Env.
    Please use namespace to avoid confusion between this function and
    brax.envs.create_fn.
    """
    return functools.partial(create, env_name, **kwargs)
