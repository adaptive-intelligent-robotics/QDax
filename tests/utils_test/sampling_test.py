import functools
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import pytest

import qdax.tasks.brax.v1 as environments
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import EnvState, Params, RNGKey
from qdax.tasks.brax.v1.env_creators import (
    scoring_function_brax_envs as scoring_function,
)
from qdax.utils.sampling import (
    average,
    closest,
    iqr,
    mad,
    median,
    mode,
    sampling,
    sampling_reproducibility,
    std,
)


def test_sampling() -> None:
    env_name = "walker2d_uni"
    episode_length = 100
    seed = 42
    policy_hidden_layer_sizes = (64, 64)
    sample_number = 32

    # Init environment
    env = environments.create(env_name, episode_length=episode_length)

    # Init a random key
    key = jax.random.key(seed)

    # Init policy network
    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of controllers
    key, subkey = jax.random.split(key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=1, axis=0)
    fake_batch = jnp.zeros(shape=(1, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # Define the function to play a step with the policy in the environment
    def play_step_fn(
        env_state: EnvState,
        policy_params: Params,
        key: RNGKey,
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

        return next_state, policy_params, key, transition

    key, subkey = jax.random.split(key)
    init_state = env.reset(subkey)

    # Prepare the scoring function
    descriptor_extraction_fn = environments.descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function,
        episode_length=episode_length,
        play_reset_fn=lambda _: init_state,
        play_step_fn=play_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
    )

    # Test function for different extractors
    def sampling_test(
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> None:

        # Compare scoring against perforing a single sample
        scoring_1_sample_fn = jax.jit(
            functools.partial(
                sampling,
                scoring_fn=scoring_fn,
                num_samples=1,
                fitness_extractor=fitness_extractor,
                descriptor_extractor=descriptor_extractor,
            )
        )

        # Evaluate individuals using the scoring functions
        fitnesses, descriptors, _ = scoring_fn(init_variables, key)
        sample_fitnesses, sample_descriptors, _ = scoring_1_sample_fn(
            init_variables, key
        )

        # Compare
        pytest.assume(
            jnp.allclose(descriptors, sample_descriptors, rtol=1e-05, atol=1e-08)
        )
        pytest.assume(jnp.allclose(fitnesses, sample_fitnesses, rtol=1e-05, atol=1e-08))

        # Compare scoring against perforing multiple samples
        scoring_multi_sample_fn = jax.jit(
            functools.partial(
                sampling,
                scoring_fn=scoring_fn,
                num_samples=sample_number,
                fitness_extractor=fitness_extractor,
                descriptor_extractor=descriptor_extractor,
            )
        )

        # Evaluate individuals using the scoring functions
        sample_fitnesses, sample_descriptors, _ = scoring_multi_sample_fn(
            init_variables, key
        )

        # Compare
        pytest.assume(
            jnp.allclose(descriptors, sample_descriptors, rtol=1e-05, atol=1e-08)
        )
        pytest.assume(jnp.allclose(fitnesses, sample_fitnesses, rtol=1e-05, atol=1e-08))

    # Call the test for each type of extractor
    sampling_test(average, average)
    sampling_test(median, median)
    sampling_test(mode, mode)
    sampling_test(closest, closest)

    # Test function for different reproducibility extractors
    def sampling_reproducibility_test(
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_repr_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        key: RNGKey,
    ) -> None:

        # Compare scoring against perforing a single sample
        scoring_1_sample_fn = jax.jit(
            functools.partial(
                sampling_reproducibility,
                scoring_fn=scoring_fn,
                num_samples=1,
                fitness_reproducibility_extractor=fitness_reproducibility_extractor,
                descriptor_reproducibility_extractor=descriptor_repr_extractor,
            )
        )

        # Evaluate individuals using the scoring functions
        key, subkey = jax.random.split(key)
        (
            _,
            _,
            _,
            fitnesses_reproducibility,
            descriptors_reproducibility,
        ) = scoring_1_sample_fn(init_variables, subkey)

        # Compare - all reproducibility should be 0
        pytest.assume(
            jnp.allclose(
                fitnesses_reproducibility,
                jnp.zeros_like(fitnesses_reproducibility),
                rtol=1e-05,
                atol=1e-05,
            )
        )
        pytest.assume(
            jnp.allclose(
                descriptors_reproducibility,
                jnp.zeros_like(descriptors_reproducibility),
                rtol=1e-05,
                atol=1e-05,
            )
        )

        # Compare scoring against perforing multiple samples
        scoring_multi_sample_fn = jax.jit(
            functools.partial(
                sampling_reproducibility,
                scoring_fn=scoring_fn,
                num_samples=sample_number,
                fitness_reproducibility_extractor=fitness_reproducibility_extractor,
                descriptor_reproducibility_extractor=descriptor_repr_extractor,
            )
        )

        # Evaluate individuals using the scoring functions
        key, subkey = jax.random.split(key)
        (
            _,
            _,
            _,
            fitnesses_reproducibility,
            descriptors_reproducibility,
        ) = scoring_multi_sample_fn(init_variables, subkey)

        # Compare - all reproducibility should be 0
        pytest.assume(
            jnp.allclose(
                fitnesses_reproducibility,
                jnp.zeros_like(fitnesses_reproducibility),
                rtol=1e-05,
                atol=1e-05,
            )
        )
        pytest.assume(
            jnp.allclose(
                descriptors_reproducibility,
                jnp.zeros_like(descriptors_reproducibility),
                rtol=1e-05,
                atol=1e-05,
            )
        )

    # Call the test for each type of extractor
    key_1, key_2, key_3 = jax.random.split(key, 3)
    sampling_reproducibility_test(std, std, key_1)
    sampling_reproducibility_test(mad, mad, key_2)
    sampling_reproducibility_test(iqr, iqr, key_3)
