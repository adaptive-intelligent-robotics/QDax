import logging
import time

import jax
import jax.numpy as jnp
from brax.training import distribution, networks
from qdax.stats.training_state import TrainingState

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


class QualityDiversityES:
    def __init__(self) -> None:
        pass

    def _eval_and_add(
        self,
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

        # this is to account for the auto-reset. we want the observation before it fell down
        first_done -= 1

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
        self,
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

        return TrainingState(
            key=key, repertoire=repertoire, metrics=metrics, state=state
        )

    def _es_one_epoch(
        self,
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

        return TrainingState(
            key=key, repertoire=repertoire, metrics=metrics, state=state
        )
