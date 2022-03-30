"""Script to define and launch a quality diversity evolution strategy on a Brax environment."""
import argparse
import functools
import os
import time

import jax
import jax.numpy as jnp
from absl import logging
from qdax.algorithms.quality_diversity import (
    QualityDiversityES,
    make_params_and_inference_fn,
)
from qdax.qd_utils import grid_archive
from qdax.stats.metrics import Metrics
from qdax.stats.saving_loading_utils import make_results_folder
from qdax.stats.timings import Timings
from qdax.stats.training_state import TrainingState
from qdax.tasks import BraxTask
from qdax.training import emitters
from qdax.training.configuration import Configuration

# see https://github.com/google/brax/blob/main/brax/experimental/composer/components/__init__.py#L43


QD_PARAMS = dict()


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


def _update_metrics(
    log_frequency: int,
    metrics: Metrics,
    epoch: int,
    repertoire: grid_archive.Repertoire,
):

    index = jnp.ceil(epoch / log_frequency).astype(int)
    scores = metrics.scores.at[index, 0].set(epoch)
    scores = scores.at[index, 1].set(repertoire.num_indivs)
    scores = scores.at[index, 2].set(jnp.nanmax(repertoire.fitness))
    scores = scores.at[index, 3].set(jnp.nansum(repertoire.fitness))
    archives = metrics.archives.at[index, :, :].set(repertoire.fitness)
    return Metrics(scores=scores, archives=archives)


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

    brax_task = BraxTask(
        env_name=args.env_name,
        episode_length=args.episode_length,
        action_repeat=1,
        num_envs=population_size,
        local_devices_to_use=local_devices_to_use,
        process_count=process_count,
    )

    num_epochs, num_evaluations = get_num_epochs_and_evaluations(
        num_epochs=parsed_arguments.num_epochs,
        num_evaluations=parsed_arguments.num_evaluations,
        population_size=parsed_arguments.batch_size,
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

    configuration = Configuration(
        args.env_name,
        num_epochs,
        parsed_arguments.episode_length,
        action_repeat=1,
        population_size=parsed_arguments.batch_size,
        seed=parsed_arguments.seed,
        log_frequency=parsed_arguments.log_frequency,
        qd_params=QD_PARAMS,
        min_bd=0.0,
        max_bd=1.0,
        grid_shape=tuple(parsed_arguments.grid_shape),
        max_devices_per_host=None,
    )

    emitter_fn = emitters.get_emitter_iso_line_dd(
        population_size=population_size,
        iso_sigma=0.005,
        line_sigma=0.05,
    )

    task = brax_task
    emitter_fn = emitter_fn
    configuration = configuration
    progress_fn = None
    experiment_name = parsed_arguments.exp_name
    result_path = results_saving_folder

    # copy paste train function

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

    # Initialize keys for random processes - need to handle for jax.
    key = jax.random.PRNGKey(seed)
    key, key_model, key_env = jax.random.split(
        key, 3
    )  # key for main training state, policy model init and train environment

    timings.init_framework = time.time() - framework_t

    # Core training environment
    env_t = (
        time.time()
    )  # NOTE: this timing doesnt work anymore at the moment - environment is initialized outside the train_fn

    key_envs = jax.random.split(key_env, local_devices_to_use)
    first_state = task.reset_fn_batch(key_envs)

    logging.info("Initialize env time: %s ", time.time() - env_t)
    timings.init_env = time.time() - env_t

    # Initialize model/policy #
    policy_t = time.time()
    # Initialize the output action distribution. action_size*2 (mean and std of a gaussian policy)
    # parametric_action_distribution = distribution.NormalTanhDistribution(
    #     event_size=core_env.action_size)
    # obs_size = core_env.observation_size
    # policy_model = make_es_model(parametric_action_distribution, obs_size)
    policy_model = task.policy_model

    # Initialize policy params - one set of params only
    policy_params = policy_model.init(key_model)

    # Initialize archive
    min_bd = configuration.min_bd
    max_bd = configuration.max_bd
    grid_shape = configuration.grid_shape

    repertoire = grid_archive.Repertoire.create(
        policy_params, min=min_bd, max=max_bd, grid_shape=grid_shape
    )

    # ============= METRICS UPDATE FN ============ #
    update_metrics_fn = functools.partial(
        _update_metrics,
        log_frequency,
    )

    # Define the Quality Diversity instance
    qdes = QualityDiversityES()

    # ============ ENVIRONMENT EVAL FUNCTIONS AND ARCHIVE ADDITION ============#
    eval_and_add_fn = jax.jit(
        functools.partial(
            qdes._eval_and_add,
            local_devices_to_use,
            task.eval,
            population_size,
            task.get_bd,
            jax.jit(repertoire.add_to_archive),
        )
    )

    # ========== INIT REPERTOIRE BY RANDOM POLICIES =============== #
    init_phase_fn = functools.partial(
        qdes._init_phase,
        task.get_random_parameters,
        population_size,
        eval_and_add_fn,
        update_metrics_fn,
    )

    # ========== ONE GENERATION/EPOCH OF ALGORITHM FN ===============#
    es_one_epoch_fn = jax.jit(
        functools.partial(
            qdes._es_one_epoch,
            emitter_fn,
            eval_and_add_fn,
            update_metrics_fn,
        )
    )

    key_debug = jax.random.PRNGKey(seed + 777)
    timings.init_policies = time.time() - policy_t

    # ================= MAIN QD ALGORITHM LOOP =================== #
    logging.info("######### START QD ALGORITHM ############")
    qd_t = time.time()

    # INIT TRAINING STATE #
    training_state = TrainingState(
        key=key,
        repertoire=repertoire,
        metrics=Metrics.create(
            log_frequency=log_frequency,
            num_epochs=num_epochs,
            grid_shape=repertoire.grid_shape,
        ),
        state=first_state,
    )
    # INIT REPERTOIRE #
    training_state = init_phase_fn(training_state)
    timings.init_QD = time.time() - qd_t

    logging.info("Starting Main QD Loop")
    # training_state = jax.lax.fori_loop(1, num_epochs+1, es_one_epoch_fn, training_state) # epoch 0 is random init ## seems to crash with large batch size on highend GPUs (e.g. RTX A6000)
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

    # ===== SAVE RESULTS AND CONFIGS ==== #
    res_dir = make_results_folder(result_path, experiment_name, configuration)
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

    inference = make_params_and_inference_fn(
        task.core_env.observation_size, task.core_env.action_size
    )

    return training_state, inference


#### util functions before launching main


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
        choices=[
            "ant",
            "hopper",
            "walker",
            "halfcheetah",
            "humanoid",
            "ant_omni",
            "humanoid_omni",
        ],
    )
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
