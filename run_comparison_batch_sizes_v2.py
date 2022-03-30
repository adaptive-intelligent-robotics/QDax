import argparse
import functools
from functools import partial
import os
import random
import sys
from typing import Callable, Optional
from absl import logging

import jax as jax
import jax.numpy as jnp

import brax
from brax.training import distribution, networks

from qdax import brax_envs
from qdax.training import qd_v2, emitters
from qdax.training.configuration import Configuration
from qdax.env_utils import play_step, generate_unroll, Transition
from qdax.types import (
    RNGKey,
)

QD_PARAMS = dict()

NUM_EVALS = 5000000
NUM_EPOCHS = 200
#EPISODE_LENGTH = 1000
LOG_FREQUENCY = 10
ACTION_REPEAT = 1

def generate_individual_model(observation_size, action_size):
  parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
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
    """Evaluate policies contained in flatten_variables in parallel"""

    # IF USING PMAP
    # pparams_device = jax.tree_map(
    #   lambda x: jnp.reshape(x, [local_devices_to_use, -1] + list(x.shape[1:])
    #                        ), pparams)

    # Perform rollouts with each policy
    unroll_fn = partial(
        generate_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        key=random_key,
    )

    _final_state, data = jax.vmap(unroll_fn, in_axes=(None, 0))(
        init_state, pparams
    )

    # create a mask to extract data properly
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=1), 0, 1)
    mask = jnp.roll(is_done, 1, axis=1)
    mask = mask.at[:, 0].set(0)

    # Scores
    fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1) #(batch_size,)
    descriptors = behavior_descriptor_extractor(data, mask) #(batch_size, bd_dim)

    # Dones
    dones = data.dones #(batch_size,)

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

def main(args):
  results_saving_folder = args.directory

  if not os.path.exists(results_saving_folder):
    raise FileNotFoundError(f"Folder {results_saving_folder} not found.")

  levels = {'fatal': logging.FATAL,
            'error': logging.ERROR,
            'warning': logging.WARNING,
            'info': logging.INFO,
            'debug': logging.DEBUG}

  logging.set_verbosity(levels[args.log_level])
  # logging.info(f"Options:\n"
  #              f"\t Log_level:{args.log_level}\n"
  #              f"\t Seed: {args.seed}\n"
  #              f"\t Batch_size:{args.batch_size}\n"
  #              f"\t Num_epochs:{args.num_epochs}\n"
  #              f"\t Episode_length:{args.episode_length}\n"
  #              f"\t Log_frequency:{args.log_frequency}\n")

  list_configurations = []
  LIST_BATCHES = args.batch_size_list
  print("BATCH SIZE LIST: ", LIST_BATCHES)

  number_replications = args.number_replications
  for _ in range(number_replications): 
  
    for population_size in LIST_BATCHES:
      seed = random.randrange(sys.maxsize)
      num_epochs = (NUM_EVALS // population_size) + 1
      #num_epochs = NUM_EPOCHS
      new_configuration = Configuration(env_name=args.env_name,
                                        num_epochs=num_epochs,
                                        episode_length=args.episode_length,
                                        action_repeat=ACTION_REPEAT,
                                        seed=seed,
                                        log_frequency=LOG_FREQUENCY,
                                        qd_params=QD_PARAMS,
                                        min_bd=float(args.min_max_bd[0]),
                                        max_bd=float(args.min_max_bd[1]),
                                        grid_shape=tuple(args.grid_shape),
                                        max_devices_per_host=None,
                                        population_size=population_size,
                                        )
      list_configurations.append(new_configuration)

  for configuration in list_configurations:
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    process_count = jax.process_count()

    emitter_fn = emitters.get_emitter_iso_line_dd(
      population_size=configuration.population_size,
      iso_sigma=0.01, # 0.01
      line_sigma=0.1, # 0.1
    )

    # create environment
    env_name = configuration.env_name
    env = brax_envs.create(env_name)

    # makes evaluation fully deterministic
    key = jax.random.PRNGKey(configuration.seed)
    key, env_key, score_key = jax.random.split(key, 3)
    reset_fn = jax.jit(env.reset)
    init_state = reset_fn(env_key)

    # define policy
    policy_model = generate_individual_model(observation_size=env.observation_size, action_size=env.action_size)

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
      experiment_name=args.exp_name,
      result_path=results_saving_folder,
      )


def process_args():
  """Read and interpret command line arguments."""
  parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('--log-level',
                      default='fatal',
                      choices=['fatal', 'error', 'warning', 'info', 'debug'])
  parser.add_argument('--exp-name', type=str, default="qdax_training") # exp name to name all results folders
  parser.add_argument('-d', '--directory', type=str, default=os.curdir) # a directory to store all results
  parser.add_argument('-n', '--number-replications', type=int, required=True)
  parser.add_argument('--env_name', type=str, required=True, choices=['ant', 'hopper', 'walker', 'halfcheetah', 'humanoid', 'ant_omni', 'humanoid_omni'])
  parser.add_argument('--grid_shape', nargs='+', type=int, required=True) # specify approrpiate grid_shape for env 
  parser.add_argument('--batch_size_list', nargs='+', type=int, default=[256, 1024, 4096, 8192, 16384]) 
  parser.add_argument('--episode_length', type=int, default=100)
  parser.add_argument('--min_max_bd', nargs='+', type=float, default=[0.0, 1.0])
  return parser.parse_args()


if __name__ == "__main__": 
  try:
    args = process_args()
    main(args)
  except Exception as e:
    logging.fatal(e, exc_info=True)
