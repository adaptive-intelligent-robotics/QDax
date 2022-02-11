import argparse
import os

import jax.numpy as jnp
# from IPython.display import HTML, clear_output
import matplotlib.pyplot as plt
from absl import logging

from qdax.tasks import BraxTask
from qdax.training.configuration import Configuration

# see https://github.com/google/brax/blob/main/brax/experimental/composer/components/__init__.py#L43

from qdax.training import qd, emitters
from qdax.training.emitters_simple.iso_dd_emitter import iso_dd_emitter, create_iso_dd_fn

import functools
from typing import Callable, Optional

from brax.envs import wrappers
from qdax.envs.unidirectional_envs import ant, walker, hopper, halfcheetah, humanoid
from qdax.envs.omnidirectional_envs import ant as ant_omni
from qdax.envs.omnidirectional_envs import humanoid as humanoid_omni
from brax.envs.env import Env

import jax

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
    raise ValueError("One of the 2 following variables should be defined: num_epochs or num_evaluations")


def main(parsed_arguments):
  results_saving_folder = parsed_arguments.directory

  if not os.path.exists(results_saving_folder):
    raise FileNotFoundError(f"Folder {results_saving_folder} not found.")

  levels = {'fatal': logging.FATAL,
            'error': logging.ERROR,
            'warning': logging.WARNING,
            'info': logging.INFO,
            'debug': logging.DEBUG}

  logging.set_verbosity(levels[parsed_arguments.log_level])

  population_size = parsed_arguments.batch_size

  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  process_count = jax.process_count()

  brax_task = BraxTask(
    env_name=args.env_name,
    episode_length=args.episode_length,
    action_repeat=1,
    num_envs=population_size,
    local_devices_to_use=local_devices_to_use,
    process_count=process_count,
  )

  num_epochs, num_evaluations = get_num_epochs_and_evaluations(num_epochs=parsed_arguments.num_epochs,
                                                               num_evaluations=parsed_arguments.num_evaluations,
                                                               population_size=parsed_arguments.batch_size)

  logging.info(f"Options:\n"
               f"\t Log_level:{parsed_arguments.log_level}\n"
               f"\t Seed: {parsed_arguments.seed}\n"
               f"\t Batch_size:{parsed_arguments.batch_size}\n"
               f"\t Num_epochs:{num_epochs}\n"
               f"\t Num_evaluations:{num_evaluations}\n"
               f"\t Episode_length:{parsed_arguments.episode_length}\n"
               f"\t Log_frequency:{parsed_arguments.log_frequency}\n")

  configuration = Configuration(args.env_name,
                                num_epochs,
                                parsed_arguments.episode_length,
                                action_repeat=1,
                                population_size=parsed_arguments.batch_size,
                                seed=parsed_arguments.seed,
                                log_frequency=parsed_arguments.log_frequency,
                                qd_params=QD_PARAMS,
                                min_bd=0.,
                                max_bd=1.,
                                grid_shape=tuple(parsed_arguments.grid_shape),
                                max_devices_per_host=None,
                                )

  emitter_fn = emitters.get_emitter_iso_line_dd(
    population_size=population_size,
    iso_sigma=0.005,
    line_sigma=0.05,
  )

  # emitter_fn = functools.partial(iso_dd_emitter, 
  #   population_size=population_size, 
  #   iso_sigma=0.01, 
  #   line_sigma=0.1)

  # emitter_fn = create_iso_dd_fn(
  #   population_size=population_size, 
  #   iso_sigma=0.01, 
  #   line_sigma=0.1)


  qd.train(
    task=brax_task,
    emitter_fn=emitter_fn,
    configuration=configuration,
    progress_fn=None,
    experiment_name=parsed_arguments.exp_name,
    result_path=results_saving_folder,
  )


def check_validity_args(parser: argparse.ArgumentParser,
                        parsed_arguments):
  num_epochs = parsed_arguments.num_epochs
  num_evaluations = parsed_arguments.num_evaluations

  if num_epochs is None and num_evaluations is None:
    parser.error("One (and only one) of the following arguments should be set: --num-epochs or --num-evaluations")
  elif num_epochs is not None and num_evaluations is not None:
    parser.error("One (and only one) of the following arguments should be set: --num-epochs or --num-evaluations")


def process_args():
  """Read and interpret command line arguments."""
  parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('--log-level',
                      default='info',
                      choices=['fatal', 'error', 'warning', 'info', 'debug'])
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--batch-size', default=2048, type=int)
  parser.add_argument('--num-epochs', default=None, type=int)
  parser.add_argument('--num-evaluations', default=None, type=int)
  parser.add_argument('--episode-length', default=100, type=int)
  parser.add_argument('--log-frequency', default=1, type=int)
  parser.add_argument('--exp-name', type=str, default="qdax_training")
  parser.add_argument('-d', '--directory', type=str, default=os.curdir)
  parser.add_argument('--env_name', type=str, required=True, choices=['ant', 'hopper', 'walker', 'halfcheetah', 'humanoid', 'ant_omni', 'humanoid_omni'])
  parser.add_argument('--grid_shape', nargs='+', type=int, required=True) # specify approrpiate grid_shape for env 

  parsed_arguments = parser.parse_args()

  check_validity_args(parser, parsed_arguments)

  return parser.parse_args()


if __name__ == "__main__":
  try:
    args = process_args()
    main(args)
  except Exception as e:
    logging.fatal(e, exc_info=True)
