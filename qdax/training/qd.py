# Author: Bryan Lim
# Email: bwl116@ic.ac.uk
#
"""
Quality-Diversity Evolution Strategy training.
"""
# TO USE MULTIPLE CPUs
# import os
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=16'

import functools
import time
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
from absl import logging
from brax.training import distribution, networks

from qdax.qd_utils import grid_archive
from qdax.stats.metrics import Metrics
from qdax.stats.saving_loading_utils import make_results_folder
from qdax.stats.timings import Timings
from qdax.stats.training_state import TrainingState
from qdax.tasks import BraxTask
from qdax.training.configuration import Configuration

# print('Jax devices: ',jax.devices())

Array = Any


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


def _eval_and_add(
    local_devices_to_use,
    evaluation_task_fn,
    population_size,
    get_bd_fn,
    add_to_archive_fn,
    training_state: TrainingState,
    pparams,
    key,
):
    key, key_es_eval = jax.random.split(key, 2)
    pparams_device = jax.tree_map(
        lambda x: jnp.reshape(x, [local_devices_to_use, -1] + list(x.shape[1:])),
        pparams,
    )

    # run evaluations - evaluate params
    eval_start_t = time.time()

    prun_es_eval = jax.pmap(evaluation_task_fn, in_axes=(0, 0, None))
    eval_scores, obs, state, done, bd, reward_trajectory = prun_es_eval(
        training_state.state, pparams_device, key_es_eval
    )

    logging.debug("Time Evaluation: %s ", time.time() - eval_start_t)

    obs = jnp.transpose(obs, axes=(1, 0, 2, 3))
    obs = jnp.reshape(obs, [1, obs.shape[0], -1, obs.shape[-1]])
    bd = jnp.transpose(bd, axes=(1, 0, 2, 3))
    bd = jnp.reshape(bd, [1, bd.shape[0], -1, bd.shape[-1]])
    eval_scores = jnp.reshape(eval_scores, (population_size, -1)).ravel()
    reward_trajectory = jnp.transpose(reward_trajectory, axes=(0, 2, 1))
    reward_trajectory = jnp.reshape(
        reward_trajectory, [1, -1, reward_trajectory.shape[-1]]
    )

    dead = jnp.zeros(population_size)
    done = jnp.transpose(done, axes=(0, 2, 1))
    done = jnp.reshape(done, [1, -1, done.shape[-1]])
    first_done = jnp.apply_along_axis(jnp.nonzero, 2, done, size=1, fill_value=-1)[
        0
    ].ravel()
    first_done -= 1  # this is to account for the auto-reset. we want the observation before it fell down

    # get fitness when first done
    eval_scores = jnp.take_along_axis(
        reward_trajectory, first_done.reshape(1, len(first_done), 1), axis=2
    ).ravel()

    # get descriptor when first done
    # bds = get_bd(obs,first_done)
    bds = get_bd_fn(bd, first_done)
    bds = jnp.transpose(bds)  # jnp.reshape(bds,(population_size, -1))

    # Update archive
    update_archive_start_t = time.time()
    repertoire = add_to_archive_fn(
        repertoire=training_state.repertoire,
        pop_p=pparams,
        bds=bds,
        eval_scores=eval_scores,
        dead=dead,
    )
    logging.debug("Time took for Adding: %s ", time.time() - update_archive_start_t)

    return repertoire, state


def _init_phase(
    get_random_parameters_fn,
    population_size,
    eval_and_add_fn,
    update_metrics_fn,
    training_state: TrainingState,
):

    logging.info(" Initialisation with random policies")
    init_start_t = time.time()
    key, key_model, key_eval = jax.random.split(training_state.key, 3)
    pparams = get_random_parameters_fn(training_state, population_size, key_model)
    logging.debug("Time Random Init: %s ", time.time() - init_start_t)

    repertoire, state = eval_and_add_fn(training_state, pparams, key_eval)
    metrics = update_metrics_fn(training_state.metrics, 0, repertoire)

    return TrainingState(key=key, repertoire=repertoire, metrics=metrics, state=state)


def _es_one_epoch(
    emitter_fn,
    eval_and_add_fn,
    update_metrics_fn,
    epoch: int,
    training_state: TrainingState,
):
    epoch_start_t = time.time()

    # generate keys for emmitter and evaluations
    key, key_emitter, key_es_eval = jax.random.split(training_state.key, 3)

    # EMITTER: SELECTION AND MUTATION #
    sel_mut_start_t = time.time()
    pparams = emitter_fn(training_state.repertoire, key_emitter)
    logging.debug("Time Selection and Mutation: %s ", time.time() - sel_mut_start_t)

    # EVALUATION #
    repertoire, state = eval_and_add_fn(training_state, pparams, key_es_eval)
    logging.debug("ES Epoch Time: %s", time.time() - epoch_start_t)

    # UPDATE METRICS #
    # metrics = jax.lax.cond((epoch+1)%log_frequency == 0 , update_metrics, lambda x:x[0], (training_state.metrics, epoch//log_frequency+1, repertoire))
    logging.debug("ES Start metrics:")
    metrics = update_metrics_fn(training_state.metrics, epoch, repertoire)
    logging.debug("ES Metrics Time: %s", time.time() - epoch_start_t)

    return TrainingState(key=key, repertoire=repertoire, metrics=metrics, state=state)


def train(
    configuration: Configuration,
    task: BraxTask,
    emitter_fn,
    experiment_name: str,
    result_path: str,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
):
    num_epochs = configuration.num_epochs
    episode_length = configuration.episode_length
    action_repeat = configuration.action_repeat
    max_devices_per_host = configuration.max_devices_per_host
    population_size = configuration.population_size
    seed = configuration.seed
    log_frequency = configuration.log_frequency
    qd_params = configuration.qd_params

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

    # ============ ENVIRONMENT EVAL FUNCTIONS AND ARCHIVE ADDITION ============#
    eval_and_add_fn = jax.jit(
        functools.partial(
            _eval_and_add,
            local_devices_to_use,
            task.eval,
            population_size,
            task.get_bd,
            jax.jit(repertoire.add_to_archive),
        )
    )

    # ========== INIT REPERTOIRE BY RANDOM POLICIES =============== #
    init_phase_fn = functools.partial(
        _init_phase,
        task.get_random_parameters,
        population_size,
        eval_and_add_fn,
        update_metrics_fn,
    )

    # ========== ONE GENERATION/EPOCH OF ALGORITHM FN ===============#
    es_one_epoch_fn = jax.jit(
        functools.partial(
            _es_one_epoch,
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


"""
The output layer here outputs parameters of the action distribution. I.e. parameters of the gaussian distibution for each action dimension
the output dim is the size of the number of params of the distribution.
"""


def make_es_model(parametric_action_distribution, obs_size):
    return networks.make_model(
        [64, 64, parametric_action_distribution.param_size], obs_size
    )


# mean of the action distribution only
def get_deterministic_actions(parameters):
    loc, scale = jnp.split(parameters, 2, axis=-1)
    act = jnp.tanh(loc)
    return act


def make_params_and_inference_fn(observation_size, action_size):
    """Creates params and inference function for the ES agent."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_model = make_es_model(parametric_action_distribution, observation_size)

    def inference_fn(params, obs, key=None):
        policy_params = params
        # action = parametric_action_distribution.sample(policy_model.apply(policy_params, obs), key)
        logits = policy_model.apply(policy_params, obs)
        action = get_deterministic_actions(logits)
        return action

    # params =  policy_model.init(jax.random.PRNGKey(0))
    return inference_fn
