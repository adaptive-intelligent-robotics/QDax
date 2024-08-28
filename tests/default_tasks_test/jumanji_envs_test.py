"""Test default rastrigin using MAP Elites"""

import functools

import jax
import jax.numpy as jnp
import jumanji
import jumanji.environments.routing.snake
import numpy as np
import pytest

from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import Descriptor, Observation
from qdax.tasks.jumanji_envs import (
    jumanji_scoring_function,
    make_policy_network_play_step_fn_jumanji,
)


def test_jumanji_utils() -> None:

    seed = 0
    policy_hidden_layer_sizes = (128, 128)
    episode_length = 200
    population_size = 100
    batch_size = population_size

    # Instantiate a Jumanji environment using the registry
    env = jumanji.make("Snake-v1")

    # Reset your (jit-able) environment
    key = jax.random.PRNGKey(0)
    state, _timestep = jax.jit(env.reset)(key)

    # Interact with the (jit-able) environment
    action = env.action_spec().generate_value()  # Action selection (dummy value here)
    state, _timestep = jax.jit(env.step)(state, action)

    # Init a random key
    random_key = jax.random.PRNGKey(seed)

    # get number of actions
    num_actions = env.action_spec().maximum + 1

    policy_layer_sizes = policy_hidden_layer_sizes + (num_actions,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jax.nn.softmax,
    )

    def observation_processing(
        observation: jumanji.environments.routing.snake.types.Observation,
    ) -> Observation:
        network_input = jnp.concatenate(
            [
                jnp.ravel(observation.grid),
                jnp.array([observation.step_count]),
                observation.action_mask.ravel(),
            ]
        )
        return network_input

    play_step_fn = make_policy_network_play_step_fn_jumanji(
        env=env,
        policy_network=policy_network,
        observation_processing=observation_processing,
    )

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=batch_size)

    # compute observation size from observation spec
    obs_spec = env.observation_spec()
    observation_size = int(
        np.prod(obs_spec.grid.shape)
        + np.prod(obs_spec.step_count.shape)
        + np.prod(obs_spec.action_mask.shape)
    )

    fake_batch = jnp.zeros(shape=(batch_size, observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # Create the initial environment states
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=batch_size, axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))

    init_states, init_timesteps = reset_fn(keys)

    # Prepare the scoring function
    def bd_extraction(
        data: QDTransition, mask: jnp.ndarray, linear_projection: jnp.ndarray
    ) -> Descriptor:
        """Extract a behavior descriptor from a trajectory.

        This extractor takes the mean observation in the trajectory and project
        it in a two dimension space.

        Args:
            data: transitions.
            mask: mask to indicate if episode is done.
            linear_projection: a linear projection.

        Returns:
            Behavior descriptors.
        """

        # pre-process the observation
        observation = jax.vmap(jax.vmap(observation_processing))(data.obs)

        # get the mean
        mean_observation = jnp.mean(observation, axis=-2)

        # project those in [-1, 1]^2
        descriptors = jnp.tanh(mean_observation @ linear_projection.T)

        return descriptors

    # create a random projection to a two dim space
    random_key, subkey = jax.random.split(random_key)
    linear_projection = jax.random.uniform(
        subkey, (2, observation_size), minval=-1, maxval=1, dtype=jnp.float32
    )

    bd_extraction_fn = functools.partial(
        bd_extraction, linear_projection=linear_projection
    )

    # define the scoring function
    scoring_fn = functools.partial(
        jumanji_scoring_function,
        init_states=init_states,
        init_timesteps=init_timesteps,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    fitnesses, descriptors, extra_scores, random_key = scoring_fn(
        init_variables, random_key
    )

    pytest.assume(fitnesses.shape == (population_size,))
    pytest.assume(jnp.sum(jnp.isnan(fitnesses)) == 0.0)
    pytest.assume(descriptors.shape == (population_size, 2))
    pytest.assume(jnp.sum(jnp.isnan(descriptors)) == 0.0)


if __name__ == "__main__":
    pytest.assume
    test_jumanji_utils()
