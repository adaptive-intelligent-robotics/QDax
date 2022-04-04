import argparse
import functools
import os
from functools import partial
from typing import Callable, Optional

import brax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from absl import logging
from brax.training import distribution, networks

from qdax import brax_envs
from qdax.env_utils import Transition, generate_unroll, play_step
from qdax.training import emitters, qd, qd_v2
from qdax.training.configuration import Configuration
from qdax.training.emitters_simple.iso_dd_emitter import (
    create_iso_dd_fn,
    iso_dd_emitter,
)
from qdax.types import RNGKey

QD_PARAMS = dict()

# QD_PARAMS = dict(bd_obs_dims = [0,1])
def get_num_epochs_and_evaluations(num_epochs, num_evaluations, population_size):
    if num_epochs is not None:
        num_evaluations = num_epochs * population_size
        return num_epochs, num_evaluations
    elif num_evaluations is not None:
        num_epochs = (num_evaluations // population_size) + 1
        return num_epochs, num_evaluations
    else:
        raise ValueError(
            "One of the 2 following variables should be defined: num_epochs or num_evaluations"
        )


def generate_individual_model(observation_size, action_size):
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    return networks.make_model(
        [64, 64, parametric_action_distribution.param_size],
        observation_size,
    )


def scoring_function(
    pparams,
    init_state: brax.envs.State,
    episode_length: int,
    random_key: RNGKey,
    play_step_fn: Callable,
    behavior_descriptor_extractor: Callable,
):

    # if we want it to be stochastic - take reset_fn as input here - already present in the main
    # use jitted reset fn to initialize a new batch of states wiht a vmap
    # env_keys = jax.random.split(random_key, population_size)
    # init_state = jax.vmap(reset_fn(env_keys))

    # Perform rollouts with each policy
    unroll_fn = partial(
        generate_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        key=random_key,
    )

    # IF USING PMAP
    # pparams_device = jax.tree_map(
    #   lambda x: jnp.reshape(x, [local_devices_to_use, -1] + list(x.shape[1:])
    #                        ), pparams)
    # data shape [batch_size, episode_length, data_dim]
    _final_state, data = jax.vmap(unroll_fn, in_axes=(None, 0))(init_state, pparams)

    # create a mask to extract data properly
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=1), 0, 1)
    mask = jnp.roll(is_done, 1, axis=1)
    mask = mask.at[:, 0].set(0)

    # Scores
    fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1)  # (batch_size,)
    descriptors = behavior_descriptor_extractor(data, mask)  # (batch_size, bd_dim)

    # Dones
    dones = data.dones  # (batch_size,)

    print(fitnesses.shape)
    print(descriptors.shape)
    print(dones.shape)

    return fitnesses, descriptors, dones, _final_state


def get_final_xy_position(data: Transition, mask: jnp.ndarray):
    """Compute final xy positon.
    This function suppose that state descriptor is the xy position, as it
    just select the final one of the state descriptors given.
    """
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    # Get behavior descriptor
    last_index = jnp.int32(jnp.sum(1.0 - mask, axis=1)) - 1
    descriptors = jax.vmap(lambda x, y: x[y])(data.state_desc, last_index)

    return descriptors.squeeze()


def main(parsed_arguments):
    results_saving_folder = parsed_arguments.directory

    if not os.path.exists(results_saving_folder):
        raise FileNotFoundError(f"Folder {results_saving_folder} not found.")

    levels = {
        "fatal": logging.FATAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }

    logging.set_verbosity(levels[parsed_arguments.log_level])

    population_size = parsed_arguments.batch_size

    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    process_count = jax.process_count()

    num_epochs, num_evaluations = get_num_epochs_and_evaluations(
        num_epochs=parsed_arguments.num_epochs,
        num_evaluations=parsed_arguments.num_evaluations,
        population_size=parsed_arguments.batch_size,
    )
    configuration = Configuration(
        args.env_name,
        num_epochs,
        parsed_arguments.episode_length,
        action_repeat=1,
        population_size=parsed_arguments.batch_size,
        seed=parsed_arguments.seed,
        log_frequency=parsed_arguments.log_frequency,
        qd_params=QD_PARAMS,
        min_bd=-15.0,
        max_bd=15.0,
        grid_shape=tuple(parsed_arguments.grid_shape),
        max_devices_per_host=None,
    )

    logging.info(
        f"Options:\n"
        f"\t Log_level:{parsed_arguments.log_level}\n"
        f"\t Seed: {parsed_arguments.seed}\n"
        f"\t Batch_size:{parsed_arguments.batch_size}\n"
        f"\t Num_epochs:{num_epochs}\n"
        f"\t Num_evaluations:{num_evaluations}\n"
        f"\t Episode_length:{parsed_arguments.episode_length}\n"
        f"\t Log_frequency:{parsed_arguments.log_frequency}\n"
    )

    emitter_fn = partial(
        iso_dd_emitter, population_size=population_size, iso_sigma=0.01, line_sigma=0.1
    )

    key = jax.random.PRNGKey(configuration.seed)
    key, env_key, score_key = jax.random.split(key, 3)

    # create environment
    env_name = configuration.env_name
    env = brax_envs.create(env_name, episode_length=configuration.episode_length)
    # makes evaluation fully deterministic
    reset_fn = jax.jit(env.reset)
    init_state = reset_fn(env_key)

    # define policy
    policy_model = generate_individual_model(
        observation_size=env.observation_size, action_size=env.action_size
    )

    # create play step function
    play_step_fn = jax.jit(partial(play_step, env=env, policy_model=policy_model))

    # create get descriptor function
    env_bd_extractor = {
        "pointmaze": get_final_xy_position,
        "ant_omni": get_final_xy_position,
    }

    # create scoring function
    scoring_fn = jax.jit(
        partial(
            scoring_function,
            init_state=init_state,
            episode_length=configuration.episode_length,
            random_key=score_key,
            play_step_fn=play_step_fn,
            behavior_descriptor_extractor=env_bd_extractor[env_name],
        )
    )

    qd_v2.train(
        scoring_fn,
        emitter_fn,
        policy_model,
        env,
        init_state,
        configuration=configuration,
        progress_fn=None,
        experiment_name=parsed_arguments.exp_name,
        result_path=results_saving_folder,
    )


def check_validity_args(parser: argparse.ArgumentParser, parsed_arguments):
    num_epochs = parsed_arguments.num_epochs
    num_evaluations = parsed_arguments.num_evaluations

    if num_epochs is None and num_evaluations is None:
        parser.error(
            "One (and only one) of the following arguments should be set: --num-epochs or --num-evaluations"
        )
    elif num_epochs is not None and num_evaluations is not None:
        parser.error(
            "One (and only one) of the following arguments should be set: --num-epochs or --num-evaluations"
        )


def process_args():
    """Read and interpret command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["fatal", "error", "warning", "info", "debug"],
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--num-epochs", default=None, type=int)
    parser.add_argument("--num-evaluations", default=None, type=int)
    parser.add_argument("--episode-length", default=100, type=int)
    parser.add_argument("--log-frequency", default=1, type=int)
    parser.add_argument("--exp-name", type=str, default="qdax_training")
    parser.add_argument("-d", "--directory", type=str, default=os.curdir)
    parser.add_argument(
        "--env_name",
        type=str,
        required=True,
    )  # choices=['ant', 'hopper', 'walker', 'halfcheetah', 'humanoid', 'ant_omni', 'humanoid_omni'])
    parser.add_argument(
        "--grid_shape", nargs="+", type=int, required=True
    )  # specify approrpiate grid_shape for env

    parsed_arguments = parser.parse_args()

    check_validity_args(parser, parsed_arguments)

    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = process_args()
        main(args)
    except Exception as e:
        logging.fatal(e, exc_info=True)
