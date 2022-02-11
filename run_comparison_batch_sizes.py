import argparse
import functools
import os
import random
import sys
from typing import Callable, Optional

# from IPython.display import HTML, clear_output
import jax as jax
from absl import logging

from brax.envs import wrappers
from brax.envs.env import Env
from qdax.envs.unidirectional_envs import ant, walker, hopper, halfcheetah, humanoid
from qdax.envs.omnidirectional_envs import ant as ant_omni
from qdax.envs.omnidirectional_envs import humanoid as humanoid_omni
from qdax.tasks import BraxTask
from qdax.training import qd, emitters
from qdax.training.configuration import Configuration


QD_PARAMS = dict()

NUM_EVALS = 5000000
NUM_EPOCHS = 200
#EPISODE_LENGTH = 1000
LOG_FREQUENCY = 10
ACTION_REPEAT = 1


def create(env_name: str,
          episode_length: int = 100,
          action_repeat: int = 1,
          auto_reset: bool = False,
          batch_size: Optional[int] = None,
          **kwargs) -> Env:
  """Creates an Env with a specified brax system."""
  
  env = {'ant': ant.QDUniAnt(**kwargs),
         'hopper': hopper.QDUniHopper(**kwargs),
         'walker': walker.QDUniWalker(**kwargs),
         'halfcheetah': halfcheetah.QDUniHalfcheetah(**kwargs),
         'humanoid': humanoid.QDUniHumanoid(**kwargs),
         'ant_omni': ant_omni.QDOmniAnt(**kwargs),
         'humanoid_omni': humanoid_omni.QDOmniHumanoid(**kwargs),
         }[env_name]

  if episode_length is not None:
    env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
  if batch_size:
    env = wrappers.VectorWrapper(env, batch_size)
  if auto_reset:
    env = wrappers.AutoResetWrapper(env)

  return env


def create_fn(**kwargs) -> Callable[..., Env]:
  """Returns a function that when called, creates an Env."""
  return functools.partial(create, **kwargs)


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

    brax_task = BraxTask(
      env_name=args.env_name,
      episode_length=args.episode_length,
      action_repeat=1,
      num_envs=configuration.population_size,
      local_devices_to_use=local_devices_to_use,
      process_count=process_count,
    )

    emitter_fn = emitters.get_emitter_iso_line_dd(
      population_size=configuration.population_size,
      iso_sigma=0.01,
      line_sigma=0.1,
    )

    qd.train(
      configuration=configuration,
      task=brax_task,
      emitter_fn=emitter_fn,
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
