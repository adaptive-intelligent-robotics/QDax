import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import pytest

from qdax import environments
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.brax_envs import scoring_function_brax_envs
from qdax.types import EnvState, Params, RNGKey
from qdax.utils.sampling import sampling


def test_sampling() -> None:
    env_name = "walker2d_uni"
    episode_length = 100
    seed = 42
    policy_hidden_layer_sizes = (64, 64)
    sample_number = 32

    # Init environment
    env = environments.create(env_name, episode_length=episode_length)

    # Init a random key
    random_key = jax.random.PRNGKey(seed)

    # Init policy network
    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=1, axis=0)
    fake_batch = jnp.zeros(shape=(1, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # Define the fonction to play a step with the policy in the environment
    def play_step_fn(
        env_state: EnvState,
        policy_params: Params,
        random_key: RNGKey,
    ) -> Tuple[EnvState, Params, RNGKey, QDTransition]:
        """
        Play an environment step and return the updated state and the transition.
        """

        actions = policy_network.apply(policy_params, env_state.obs)

        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return next_state, policy_params, random_key, transition

    # Create the initial environment states for samples and final indivs
    reset_fn = jax.jit(jax.vmap(env.reset))
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=1, axis=0)
    init_states = reset_fn(keys)

    # Compare scoring against perforing a single sample
    bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function_brax_envs,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )
    scoring_1_sample_fn = functools.partial(
        sampling,
        scoring_fn=scoring_fn,
        num_samples=1,
    )

    # Evaluate individuals using the scoring functions
    fitnesses, descriptors, _, _ = scoring_fn(init_variables, random_key)
    sample_fitnesses, sample_descriptors, _, _ = scoring_1_sample_fn(
        init_variables, random_key
    )

    # Compare
    pytest.assume(jnp.allclose(descriptors, sample_descriptors, rtol=1e-05, atol=1e-08))
    pytest.assume(jnp.allclose(fitnesses, sample_fitnesses, rtol=1e-05, atol=1e-08))

    # Compare scoring against perforing multiple samples
    scoring_multi_sample_fn = functools.partial(
        sampling,
        scoring_fn=scoring_fn,
        num_samples=sample_number,
    )

    # Evaluate individuals using the scoring functions
    sample_fitnesses, sample_descriptors, _, _ = scoring_multi_sample_fn(
        init_variables, random_key
    )

    # Compare
    pytest.assume(jnp.allclose(descriptors, sample_descriptors, rtol=1e-05, atol=1e-08))
    pytest.assume(jnp.allclose(fitnesses, sample_fitnesses, rtol=1e-05, atol=1e-08))
