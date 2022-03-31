import argparse
import functools
import os
import time
from functools import partial
from typing import Callable, Optional

import brax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from absl import logging
from brax.training import distribution, networks
from qdax import brax_envs
from qdax.algorithms.quality_diversity import (
    QualityDiversityES,
    make_params_and_inference_fn,
)
from qdax.env_utils import Transition, generate_unroll, play_step
from qdax.qd_utils.grid_repertoire import GridRepertoire
from qdax.stats.metrics import Metrics
from qdax.stats.saving_loading_utils import make_results_folder
from qdax.stats.timings import Timings
from qdax.stats.training_state import TrainingState
from qdax.training import emitters, qd, qd_v2
from qdax.training.configuration import Configuration
from qdax.training.emitters_simple.iso_dd_emitter import (
    create_iso_dd_fn,
    iso_dd_emitter,
)
from qdax.types import RNGKey

QD_PARAMS = dict()


def _update_metrics(
    log_frequency: int,
    metrics: Metrics,
    epoch: int,
    repertoire: GridRepertoire,
):

    index = jnp.ceil(epoch / log_frequency).astype(int)
    scores = metrics.scores.at[index, 0].set(epoch)
    scores = scores.at[index, 1].set(repertoire.num_indivs)
    scores = scores.at[index, 2].set(jnp.nanmax(repertoire.fitness))
    scores = scores.at[index, 3].set(jnp.nansum(repertoire.fitness))
    archives = metrics.archives.at[index, :, :].set(repertoire.fitness)
    return Metrics(scores=scores, archives=archives)


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


def get_random_parameters(
    policy_model, training_state: TrainingState, population_size: int, key_model
):
    t = time.time()
    key_model_batch = jax.random.split(key_model, population_size)
    policy_params_fun = jax.vmap(policy_model.init)
    pparams = policy_params_fun(key_model_batch)
    logging.info("Init policies  %s ", time.time() - t)
    return pparams


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
        env_name=args.env_name,
        num_epochs=num_epochs,
        episode_length=parsed_arguments.episode_length,
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

    # beginning of the previous train function
    progress_fn = None

    num_epochs = configuration.num_epochs
    max_devices_per_host = configuration.max_devices_per_host
    population_size = configuration.population_size
    seed = configuration.seed
    log_frequency = configuration.log_frequency

    timings = Timings(log_frequency=log_frequency, num_epochs=num_epochs)
    start_t = time.time()
    framework_t = time.time()

    # INIT FRAMEWORK #
    # Initialization of env parameters and devices #
    num_envs = population_size  #
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, "
        "devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    logging.info("Local devices to use: %d ", local_devices_to_use)
    logging.info(
        "Batch size on 1 device for env: %d",
        num_envs // local_devices_to_use // process_count,
    )

    # Initialize keys for random processes - need to handle for jax
    # key for main training state, policy model init and train environment
    key = jax.random.PRNGKey(seed)
    key, key_model = jax.random.split(key)

    timings.init_framework = time.time() - framework_t

    # Core training environment
    env_t = (
        time.time()
    )  # NOTE: this timing doesnt work anymore at the moment - environment is initialized outside the train_fn
    logging.info("Initialize env time: %s ", time.time() - env_t)
    timings.init_env = time.time() - env_t

    # Initialize archive
    min_bd = configuration.min_bd
    max_bd = configuration.max_bd
    grid_shape = configuration.grid_shape

    # Init policy params
    policy_params = policy_model.init(key_model)

    # create repertoire
    repertoire = GridRepertoire.create(
        policy_params, min=min_bd, max=max_bd, grid_shape=grid_shape
    )

    # fucntion to manage metrics
    update_metrics_fn = jax.jit(
        functools.partial(
            _update_metrics,
            log_frequency,
        )
    )

    # init qdes instance
    qdes = QualityDiversityES(
        scoring_fn=scoring_fn, update_metrics_fn=update_metrics_fn
    )

    # init repertoire with random policies
    get_random_params_fn = functools.partial(get_random_parameters, policy_model)
    init_phase_fn = functools.partial(
        qdes._init_phase,
        get_random_params_fn,
        population_size,
    )

    # one epoch of the algorithm
    es_one_epoch_fn = jax.jit(
        functools.partial(
            qdes._es_one_epoch,
            emitter_fn,
        )
    )

    # main loop
    logging.info("######### START QD ALGORITHM ############")
    qd_t = time.time()

    # init training state
    training_state = TrainingState(
        key=key,
        repertoire=repertoire,
        metrics=Metrics.create(
            log_frequency=log_frequency,
            num_epochs=num_epochs,
            grid_shape=repertoire.grid_shape,
        ),
        state=init_state,
    )
    # init repertoire
    training_state = init_phase_fn(training_state)
    timings.init_QD = time.time() - qd_t

    logging.info("Starting Main QD Loop")

    for i in range(1, num_epochs + 1):
        epoch_t = time.time()
        training_state = es_one_epoch_fn(i, training_state)

        # log timings
        logging.debug("epoch loop Time: %s ", time.time() - epoch_t)
        epoch_duration = time.time() - epoch_t
        epoch_runtime = time.time() - start_t
        timings.avg_epoch = ((i - 1) * timings.avg_epoch + (epoch_duration)) / float(i)
        timings.avg_eval_per_sec = (
            (i - 1) * timings.avg_eval_per_sec + population_size / epoch_duration
        ) / float(i)

        index = jnp.ceil(i / log_frequency).astype(int)
        timings.epoch_runtime = timings.epoch_runtime.at[index, 0].set(epoch_runtime)
        # print("Index: ",index, epoch_runtime)

    timings.full_training = time.time() - start_t
    logging.info(timings)
    # training_state.repertoire.fitness.block_until_ready()
    logging.debug("Total main loop Time: %s ", time.time() - start_t)
    logging.info("Repertoire size: %d ", training_state.repertoire.num_indivs)
    logging.info(
        "Scores [epoch, num_indivs, best fitness, QD score]:\n %s ",
        training_state.metrics.scores,
    )

    # save results and configs
    res_dir = make_results_folder(
        results_saving_folder, parsed_arguments.exp_name, configuration
    )
    logging.info("Saving results in %s ", res_dir)
    configuration.save_to_json(folder=res_dir)
    timings.save(folder=res_dir)
    training_state.save(folder=res_dir)
    training_state.metrics.save(folder=res_dir)

    if progress_fn:
        metrics = dict(
            **dict(
                {
                    "train/generation": num_epochs + 1,
                    "repertoire/repertoire_size": training_state.repertoire.num_indivs,
                }
            )
        )
        progress_fn(metrics, training_state.repertoire)

    inference = make_params_and_inference_fn(env.observation_size, env.action_size)

    return training_state, inference


# Function necessary to launch this script - handle args


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
