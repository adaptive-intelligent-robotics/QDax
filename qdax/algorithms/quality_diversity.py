import logging
import time
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from brax.training import distribution, networks
from qdax.stats.training_state import TrainingState
from qdax.types import RNGKey

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
    def __init__(self, scoring_fn: Callable, update_metrics_fn: Callable) -> None:
        self._scoring_fn = scoring_fn
        self._update_metrics_fn = update_metrics_fn

    @partial(jax.jit, static_argnames=("self"))
    def _eval_and_add(
        self,
        training_state: TrainingState,
        pparams: Any,
    ):

        # run evaluations - evaluate params
        eval_start_t = time.time()
        eval_scores, bds, _dones, state = self._scoring_fn(pparams)

        logging.debug("Time Evaluation: %s ", time.time() - eval_start_t)

        dead = jnp.zeros(bds.shape[0])

        # Update archive
        update_archive_start_t = time.time()
        repertoire = training_state.repertoire.add_to_archive(
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
        training_state: TrainingState,
    ):

        logging.info(" Initialisation with random policies")
        init_start_t = time.time()
        key, key_model = jax.random.split(training_state.key)
        pparams = get_random_parameters_fn(training_state, population_size, key_model)
        logging.debug("Time Random Init: %s ", time.time() - init_start_t)

        repertoire, state = self._eval_and_add(
            training_state=training_state, pparams=pparams
        )
        metrics = self._update_metrics_fn(training_state.metrics, 0, repertoire)

        return TrainingState(
            key=key, repertoire=repertoire, metrics=metrics, state=state
        )

    @partial(jax.jit, static_argnames=("self", "emitter_fn"))
    def _es_one_epoch(
        self,
        emitter_fn,
        epoch: int,
        training_state: TrainingState,
    ):
        epoch_start_t = time.time()

        # generate keys for emmitter and evaluations
        key, key_emitter = jax.random.split(training_state.key)

        # emitter: selection and mutation
        sel_mut_start_t = time.time()
        pparams = emitter_fn(training_state.repertoire, key_emitter)
        logging.debug("Time Selection and Mutation: %s ", time.time() - sel_mut_start_t)

        # evaluation
        repertoire, state = self._eval_and_add(
            training_state=training_state, pparams=pparams
        )
        logging.debug("ES Epoch Time: %s", time.time() - epoch_start_t)

        # update metrics
        logging.debug("ES Start metrics:")
        metrics = self._update_metrics_fn(training_state.metrics, epoch, repertoire)
        logging.debug("ES Metrics Time: %s", time.time() - epoch_start_t)

        return TrainingState(
            key=key, repertoire=repertoire, metrics=metrics, state=state
        )
