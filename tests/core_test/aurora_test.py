"""Tests AURORA implementation"""

import functools
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import pytest

from qdax import environments
from qdax.core.aurora import AURORA
from qdax.core.containers.unstructured_repertoire import UnstructuredRepertoire
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.environments.bd_extractors import get_aurora_encoding
from qdax.tasks.brax_envs import get_aurora_scoring_fn, create_default_brax_task_components
from qdax.types import EnvState, Params, RNGKey, Observation
from qdax.utils import train_seq2seq


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
    random_key = jax.random.PRNGKey(seed)

    # Init environment
    env, policy_network, scoring_fn, random_key = create_default_brax_task_components(
        env_name=env_name,
        random_key=random_key,
    )

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
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

        print("State Observations: ", state_obs)
        print("XY positions: ", state_desc)

        if observation_option == "full":
            observations = jnp.concatenate([state_desc, state_obs], axis=-1)
            print("New observations: ", observations)
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
    variation_fn = functools.partial(isoline_variation, iso_sigma=0.05, line_sigma=0.1)
    mixing_emitter = MixingEmitter(
        mutation_fn=lambda x, y: (x, y),
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name]

    # Define a metrics function
    def metrics_fn(repertoire: UnstructuredRepertoire) -> Dict:

        # Get metrics
        grid_empty = repertoire.fitnesses == -jnp.inf
        qd_score = jnp.sum(repertoire.fitnesses, where=~grid_empty)
        # Add offset for positive qd_score
        qd_score += reward_offset * episode_length * jnp.sum(1.0 - grid_empty)
        coverage = 100 * jnp.mean(1.0 - grid_empty)
        max_fitness = jnp.max(repertoire.fitnesses)

        return {"qd_score": qd_score, "max_fitness": max_fitness, "coverage": coverage}

    # Init algorithm
    # AutoEncoder Params and INIT
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

    # define the seq2seq model
    model = train_seq2seq.get_model(
        observations_dims[-1], True, hidden_size=hidden_size
    )

    encoder_fn = functools.partial(
        get_aurora_encoding,
        model=model,
    )

    # Instantiate AURORA
    aurora = AURORA(
        scoring_function=aurora_scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_fn,
        encoder_function=encoder_fn,
    )

    @jax.jit
    def update_scan_fn(carry: Any, _: Any) -> Any:
        """Scan the udpate function."""
        # TODO: fix shadowing names from outer scopes.
        (
            repertoire,
            random_key,
            model_params,
            mean_observations,
            std_observations,
        ) = carry

        # update
        (repertoire, _, metrics, random_key,) = aurora.update(
            repertoire,
            None,
            random_key,
            model_params,
            mean_observations,
            std_observations,
        )

        return (
            (repertoire, random_key, model_params, mean_observations, std_observations),
            metrics,
        )




    # init the model params
    random_key, subkey = jax.random.split(random_key)
    model_params = train_seq2seq.get_initial_params(
        model, subkey, (1, *observations_dims)
    )

    print(jax.tree_map(lambda x: x.shape, model_params))

    # define arbitrary observation's mean/std
    mean_observations = jnp.zeros(observations_dims[-1])
    std_observations = jnp.ones(observations_dims[-1])

    # init step of the aurora algorithm
    repertoire, _, random_key = aurora.init(
        init_variables,
        random_key,
        model_params,
        mean_observations,
        std_observations,
        jnp.array(l_value_init),
        max_size,
    )

    # initializing means and stds and AURORA
    random_key, subkey = jax.random.split(random_key)
    model_params, mean_observations, std_observations = train_seq2seq.lstm_ae_train(
        subkey,
        repertoire,
        model_params,
        0,
        hidden_size=hidden_size,
        batch_size=lstm_batch_size,
    )

    # design aurora's schedule
    default_update_base = 10
    update_base = int(jnp.ceil(default_update_base / log_freq))
    schedules = jnp.cumsum(jnp.arange(update_base, 1000, update_base))

    current_step_estimation = 0

    # Main loop
    n_target = 1024

    previous_error = jnp.sum(repertoire.fitnesses != -jnp.inf) - n_target

    iteration = 0
    while iteration < max_iterations:

        (
            (repertoire, random_key, model_params, mean_observations, std_observations),
            metrics,
        ) = jax.lax.scan(
            update_scan_fn,
            (repertoire, random_key, model_params, mean_observations, std_observations),
            (),
            length=log_freq,
        )

        # update nb steps estimation
        current_step_estimation += batch_size * episode_length * log_freq

        # autoencoder steps and CVC
        if (iteration + 1) in schedules:
            # train the autoencoder
            random_key, subkey = jax.random.split(random_key)
            (
                model_params,
                mean_observations,
                std_observations,
            ) = train_seq2seq.lstm_ae_train(
                subkey,
                repertoire,
                model_params,
                iteration,
                hidden_size=hidden_size,
                batch_size=lstm_batch_size,
            )

            # re-addition of all the new behavioural descriotpors with the new ae
            normalized_observations = (
                repertoire.observations - mean_observations
            ) / std_observations

            new_descriptors = model.apply(
                {"params": model_params}, normalized_observations, method=model.encode
            )
            repertoire = repertoire.init(
                genotypes=repertoire.genotypes,
                fitnesses=repertoire.fitnesses,
                descriptors=new_descriptors,
                observations=repertoire.observations,
                l_value=repertoire.l_value,
                max_size=repertoire.max_size,
            )
            num_indivs = jnp.sum(repertoire.fitnesses != -jnp.inf)

        elif iteration % 2 == 0:
            # update the l value
            num_indivs = jnp.sum(repertoire.fitnesses != -jnp.inf)

            # CVC Implementation to keep a constant number of individuals in the archive
            current_error = num_indivs - n_target
            change_rate = current_error - previous_error
            prop_gain = 1 * 10e-6
            l_value = (
                repertoire.l_value
                + (prop_gain * current_error)
                + (prop_gain * change_rate)
            )

            previous_error = current_error

            repertoire = repertoire.init(
                genotypes=repertoire.genotypes,
                fitnesses=repertoire.fitnesses,
                descriptors=repertoire.descriptors,
                observations=repertoire.observations,
                l_value=l_value,
                max_size=repertoire.max_size,
            )

        iteration += 1

    pytest.assume(repertoire is not None)


if __name__ == "__main__":
    test_aurora(env_name="pointmaze", batch_size=10)
