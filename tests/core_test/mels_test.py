"""Tests MAP-Elites Low-Spread implementation."""

import functools
from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import pytest

import qdax.tasks.brax.v1 as environments
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.containers.mels_repertoire import MELSRepertoire
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.mels import MELS
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import EnvState, Params, RNGKey
from qdax.tasks.brax.v1.env_creators import scoring_function_brax_envs
from qdax.utils.sampling import multi_sample_scoring_function


@pytest.mark.parametrize(
    "env_name, batch_size, custom_repertoire",
    [
        ("walker2d_uni", 1, False),
        ("walker2d_uni", 10, False),
        ("hopper_uni", 10, False),
        ("walker2d_uni", 1, True),
        ("walker2d_uni", 10, True),
        ("hopper_uni", 10, True),
    ],
)
def test_mels(env_name: str, batch_size: int, custom_repertoire: bool) -> None:
    batch_size = batch_size
    env_name = env_name
    num_samples = 5
    episode_length = 100
    num_iterations = 5
    seed = 42
    policy_hidden_layer_sizes = (64, 64)
    num_init_cvt_samples = 1000
    num_centroids = 50
    min_descriptor = 0.0
    max_descriptor = 1.0

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

    # Init population of controllers. There are batch_size controllers, and each
    # controller will be evaluated num_samples times.
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch = jnp.zeros(shape=(batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # Define the function to play a step with the policy in the environment
    def play_step_fn(
        env_state: EnvState,
        policy_params: Params,
        key: RNGKey,
    ) -> Tuple[EnvState, Params, RNGKey, QDTransition]:
        """Play an environment step and return the updated state and the
        transition."""

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

    # Prepare the scoring function
    descriptor_extraction_fn = environments.descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function_brax_envs,
        episode_length=episode_length,
        play_reset_fn=env.reset,
        play_step_fn=play_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
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
    def metrics_fn(repertoire: MELSRepertoire) -> Dict:
        # Get metrics
        grid_empty = repertoire.fitnesses == -jnp.inf
        qd_score = jnp.sum(repertoire.fitnesses, where=~grid_empty)
        # Add offset for positive qd_score
        qd_score += reward_offset * episode_length * jnp.sum(1.0 - grid_empty)
        coverage = 100 * jnp.mean(1.0 - grid_empty)
        max_fitness = jnp.max(repertoire.fitnesses)

        return {"qd_score": qd_score, "max_fitness": max_fitness, "coverage": coverage}

    # Instantiate ME-LS.
    if custom_repertoire:
        scoring_fn = partial(
            multi_sample_scoring_function,
            scoring_fn=scoring_fn,
            num_samples=num_samples,
        )
        mels = MAPElites(
            scoring_function=scoring_fn,
            emitter=mixing_emitter,
            metrics_function=metrics_fn,
            repertoire_init=MELSRepertoire.init,
        )
    else:
        # Instantiate ME-LS.
        mels = MELS(
            scoring_function=scoring_fn,
            emitter=mixing_emitter,
            metrics_function=metrics_fn,
            num_samples=num_samples,
        )

    # Compute the centroids
    key, subkey = jax.random.split(key)
    centroids = compute_cvt_centroids(
        num_descriptors=env.descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_descriptor,
        maxval=max_descriptor,
        key=subkey,
    )

    # Compute initial repertoire
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = mels.init(
        init_variables, centroids, subkey
    )

    # Run the algorithm
    (
        repertoire,
        emitter_state,
        key,
    ), metrics = jax.lax.scan(
        mels.scan_update,
        (repertoire, emitter_state, key),
        (),
        length=num_iterations,
    )

    pytest.assume(repertoire is not None)


@pytest.mark.parametrize(
    "env_name, batch_size, custom_repertoire",
    [
        ("walker2d_uni", 1, False),
        ("walker2d_uni", 10, False),
        ("hopper_uni", 10, False),
        ("walker2d_uni", 1, True),
        ("walker2d_uni", 10, True),
        ("hopper_uni", 10, True),
    ],
)
def test_mels_ask_tell(env_name: str, batch_size: int, custom_repertoire: bool) -> None:
    batch_size = batch_size
    env_name = env_name
    num_samples = 5
    episode_length = 100
    num_iterations = 5
    seed = 42
    policy_hidden_layer_sizes = (64, 64)
    num_init_cvt_samples = 1000
    num_centroids = 50
    min_descriptor = 0.0
    max_descriptor = 1.0

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

    # Init population of controllers. There are batch_size controllers, and each
    # controller will be evaluated num_samples times.
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch = jnp.zeros(shape=(batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # Define the function to play a step with the policy in the environment
    def play_step_fn(
        env_state: EnvState,
        policy_params: Params,
        key: RNGKey,
    ) -> Tuple[EnvState, Params, RNGKey, QDTransition]:
        """Play an environment step and return the updated state and the
        transition."""

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

    # Prepare the scoring function
    descriptor_extraction_fn = environments.descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function_brax_envs,
        episode_length=episode_length,
        play_reset_fn=env.reset,
        play_step_fn=play_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
    )
    scoring_fn = partial(
        multi_sample_scoring_function,
        scoring_fn=scoring_fn,
        num_samples=num_samples,
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
    def metrics_fn(repertoire: MELSRepertoire) -> Dict:
        # Get metrics
        grid_empty = repertoire.fitnesses == -jnp.inf
        qd_score = jnp.sum(repertoire.fitnesses, where=~grid_empty)
        # Add offset for positive qd_score
        qd_score += reward_offset * episode_length * jnp.sum(1.0 - grid_empty)
        coverage = 100 * jnp.mean(1.0 - grid_empty)
        max_fitness = jnp.max(repertoire.fitnesses)

        return {"qd_score": qd_score, "max_fitness": max_fitness, "coverage": coverage}

    # Instantiate ME-LS.
    if custom_repertoire:
        mels = MAPElites(
            scoring_function=scoring_fn,
            emitter=mixing_emitter,
            metrics_function=metrics_fn,
            repertoire_init=MELSRepertoire.init,
        )
    else:
        # Instantiate ME-LS.
        mels = MELS(
            scoring_function=scoring_fn,
            emitter=mixing_emitter,
            metrics_function=metrics_fn,
            num_samples=num_samples,
        )

    # Compute the centroids
    key, subkey = jax.random.split(key)
    centroids = compute_cvt_centroids(
        num_descriptors=env.descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_descriptor,
        maxval=max_descriptor,
        key=subkey,
    )

    # Evaluate the initial population
    key, subkey = jax.random.split(key)
    fitnesses, descriptors, extra_scores = scoring_fn(init_variables, subkey)

    # Compute initial repertoire and emitter state
    repertoire, emitter_state, init_metrics = mels.init_ask_tell(
        genotypes=init_variables,
        fitnesses=fitnesses,
        descriptors=descriptors,
        centroids=centroids,
        key=key,
        extra_scores=extra_scores,
    )

    # Run the algorithm
    for _ in range(num_iterations):
        key, subkey = jax.random.split(key)
        # Generate solutions
        genotypes, extra_info = mels.ask(repertoire, emitter_state, subkey)

        # Evaluate solutions: get fitness, descriptor and extra scores.
        # This is where custom evaluations on CPU or GPU can be added.
        key, subkey = jax.random.split(key)
        fitnesses, descriptors, extra_scores = scoring_fn(genotypes, subkey)

        # Update MAP-Elites
        repertoire, emitter_state, current_metrics = mels.tell(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            repertoire=repertoire,
            emitter_state=emitter_state,
            extra_scores=extra_scores,
            extra_info=extra_info,
        )

    pytest.assume(repertoire is not None)


if __name__ == "__main__":
    test_mels(env_name="pointmaze", batch_size=10)
    test_mels_ask_tell(env_name="pointmaze", batch_size=10)
