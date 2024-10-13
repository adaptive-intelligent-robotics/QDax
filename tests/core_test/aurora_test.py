"""Tests AURORA implementation"""

import functools
from typing import Tuple

import brax.envs
import jax
import jax.numpy as jnp
import pytest

from qdax import environments
from qdax.core.aurora import AURORA
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.custom_types import Observation
from qdax.environments.descriptor_extractors import (
    AuroraExtraInfoNormalization,
    get_aurora_encoding,
)
from qdax.tasks.brax_envs import (
    create_default_brax_task_components,
    get_aurora_scoring_fn,
)
from qdax.utils import train_seq2seq
from qdax.utils.metrics import default_qd_metrics
from tests.core_test.map_elites_test import get_mixing_emitter


def get_observation_dims(
    observation_option: str,
    env: brax.envs.Env,
    max_observation_size: int,
    episode_length: int,
    traj_sampling_freq: int,
    prior_descriptor_dim: int,
) -> Tuple[int, int]:
    obs_dim = jnp.minimum(env.observation_size, max_observation_size)
    if observation_option == "full":
        observations_dims = (
            episode_length // traj_sampling_freq,
            obs_dim + prior_descriptor_dim,
        )
    elif observation_option == "no_sd":
        observations_dims = (
            episode_length // traj_sampling_freq,
            obs_dim,
        )
    elif observation_option == "only_sd":
        observations_dims = (episode_length // traj_sampling_freq, prior_descriptor_dim)
    else:
        raise ValueError(f"Unknown observation option: {observation_option}")

    return observations_dims


@pytest.mark.parametrize(
    "env_name, batch_size",
    [("halfcheetah_uni", 10), ("walker2d_uni", 10), ("hopper_uni", 10)],
)
def test_aurora(env_name: str, batch_size: int) -> None:
    episode_length = 250
    max_iterations = 5
    seed = 42
    max_size = 50

    lstm_batch_size = 12

    observation_option = "no_sd"  # "full", "no_sd", "only_sd"
    hidden_size = 5
    l_value_init = 0.2

    traj_sampling_freq = 10
    max_observation_size = 25
    prior_descriptor_dim = 2

    log_freq = 5

    # Init a random key
    key = jax.random.key(seed)

    # Init environment
    key, subkey = jax.random.split(key)
    env, policy_network, scoring_fn = create_default_brax_task_components(
        env_name=env_name,
        key=subkey,
    )

    # Init population of controllers
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch = jnp.zeros(shape=(batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    def observation_extractor_fn(
        data: QDTransition,
    ) -> Observation:
        """Extract observation from the state."""
        state_obs = data.obs[:, ::traj_sampling_freq, :max_observation_size]

        # add the x/y position - (batch_size, traj_length, 2)
        state_desc = data.state_desc[:, ::traj_sampling_freq]

        if observation_option == "full":
            observations = jnp.concatenate([state_desc, state_obs], axis=-1)
        elif observation_option == "no_sd":
            observations = state_obs
        elif observation_option == "only_sd":
            observations = state_desc
        else:
            raise ValueError("Unknown observation option.")

        return observations

    # Prepare the scoring function
    aurora_scoring_fn = get_aurora_scoring_fn(
        scoring_fn=scoring_fn,
        observation_extractor_fn=observation_extractor_fn,
    )

    # Define emitter
    mixing_emitter = get_mixing_emitter(batch_size)

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name]

    # Define a metrics function
    metrics_fn = functools.partial(default_qd_metrics, qd_offset=reward_offset)

    # Init algorithm
    # AutoEncoder Params and INIT
    observations_dims = get_observation_dims(
        observation_option=observation_option,
        env=env,
        max_observation_size=max_observation_size,
        episode_length=episode_length,
        traj_sampling_freq=traj_sampling_freq,
        prior_descriptor_dim=prior_descriptor_dim,
    )

    # define the seq2seq model
    model = train_seq2seq.get_model(
        int(observations_dims[-1]), True, hidden_size=hidden_size
    )

    # define the encoder function
    encoder_fn = jax.jit(
        functools.partial(
            get_aurora_encoding,
            model=model,
        )
    )

    # define the training function
    train_fn = functools.partial(
        train_seq2seq.lstm_ae_train,
        model=model,
        batch_size=lstm_batch_size,
    )

    # Instantiate AURORA algorithm
    aurora = AURORA(
        scoring_function=aurora_scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_fn,
        encoder_function=encoder_fn,
        training_function=train_fn,
    )

    # init the model params
    key, subkey = jax.random.split(key)
    model_params = train_seq2seq.get_initial_params(
        model, subkey, (1, *observations_dims)
    )

    # define arbitrary observation's mean/std
    mean_observations = jnp.zeros(observations_dims[-1])
    std_observations = jnp.ones(observations_dims[-1])

    # init all the information needed by AURORA to compute encodings
    aurora_extra_info = AuroraExtraInfoNormalization.create(
        model_params,
        mean_observations,
        std_observations,
    )

    # init step of the aurora algorithm
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics, aurora_extra_info = aurora.init(
        init_variables,
        aurora_extra_info,
        jnp.asarray(l_value_init),
        max_size,
        subkey,
    )

    # initializing means and stds and AURORA
    key, subkey = jax.random.split(key)
    repertoire, aurora_extra_info = aurora.train(
        repertoire, model_params, iteration=0, key=subkey
    )

    # design aurora's schedule
    default_update_base = 10
    update_base = int(jnp.ceil(default_update_base / log_freq))
    schedules = jnp.cumsum(jnp.arange(update_base, 1000, update_base))

    current_step_estimation = 0

    ############################
    # Main loop
    ############################

    target_repertoire_size = 1024

    previous_error = jnp.sum(repertoire.fitnesses != -jnp.inf) - target_repertoire_size

    iteration = 0
    while iteration < max_iterations:
        # standard MAP-Elites-like loop
        for _ in range(log_freq):
            key, subkey = jax.random.split(key)
            repertoire, emitter_state, _ = aurora.update(
                repertoire,
                emitter_state,
                subkey,
                aurora_extra_info=aurora_extra_info,
            )

        # update nb steps estimation
        current_step_estimation += batch_size * episode_length * log_freq

        # autoencoder steps and Container Size Control (CSC)
        if (iteration + 1) in schedules:
            # train the autoencoder (includes the CSC)
            key, subkey = jax.random.split(key)
            repertoire, aurora_extra_info = aurora.train(
                repertoire, model_params, iteration, subkey
            )

        elif iteration % 2 == 0:
            # only CSC
            repertoire, previous_error = aurora.container_size_control(
                repertoire,
                target_size=target_repertoire_size,
                previous_error=previous_error,
            )

        iteration += 1

    pytest.assume(repertoire is not None)


if __name__ == "__main__":
    test_aurora(env_name="pointmaze", batch_size=10)
