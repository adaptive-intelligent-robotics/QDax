import functools
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import pytest

from qdax import environments
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from qdax.core.emitters.mees_emitter import MEESConfig, MEESEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import EnvState, Params, RNGKey
from qdax.tasks.brax_envs import scoring_function_brax_envs as scoring_function


def test_mees() -> None:
    env_name = "walker2d_uni"
    episode_length = 100
    num_iterations = 10
    seed = 42
    policy_hidden_layer_sizes = (64, 64)
    num_init_cvt_samples = 1000
    num_centroids = 50
    min_descriptor = 0.0
    max_descriptor = 1.0

    # MEES Emitter params
    sample_number = 128
    sample_sigma = 0.02
    num_optimizer_steps = 2
    learning_rate = 0.01
    l2_coefficient = 0.02
    novelty_nearest_neighbors = 10

    adam_optimizer = True
    sample_mirror = True
    sample_rank_norm = True
    use_explore = True
    use_exploit = True

    # Init environment
    env = environments.create(env_name, episode_length=episode_length)
    reset_fn = jax.jit(env.reset)

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

    # Prepare the scoring function
    descriptor_extraction_fn = environments.descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function,
        episode_length=episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name]

    # Define a metrics function
    def metrics_function(repertoire: MapElitesRepertoire) -> Dict:

        # Get metrics
        grid_empty = repertoire.fitnesses == -jnp.inf
        qd_score = jnp.sum(repertoire.fitnesses, where=~grid_empty)
        # Add offset for positive qd_score
        qd_score += reward_offset * episode_length * jnp.sum(1.0 - grid_empty)
        coverage = 100 * jnp.mean(1.0 - grid_empty)
        max_fitness = jnp.max(repertoire.fitnesses)

        return {"qd_score": qd_score, "max_fitness": max_fitness, "coverage": coverage}

    # Define the MEES-emitter config
    mees_emitter_config = MEESConfig(
        sample_number=sample_number,
        sample_sigma=sample_sigma,
        sample_mirror=sample_mirror,
        sample_rank_norm=sample_rank_norm,
        num_optimizer_steps=num_optimizer_steps,
        adam_optimizer=adam_optimizer,
        learning_rate=learning_rate,
        l2_coefficient=l2_coefficient,
        novelty_nearest_neighbors=novelty_nearest_neighbors,
        use_explore=use_explore,
        use_exploit=use_exploit,
    )

    # Get the emitter
    mees_emitter = MEESEmitter(
        config=mees_emitter_config,
        total_generations=num_iterations,
        scoring_fn=scoring_fn,
        num_descriptors=env.descriptor_length,
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

    # Instantiate MAP Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=mees_emitter,
        metrics_function=metrics_function,
    )

    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = map_elites.init(
        init_variables, centroids, subkey
    )

    @jax.jit
    def update_scan_fn(carry: Any, _: Any) -> Any:
        # iterate over grid
        repertoire, emitter_state, key = carry
        key, subkey = jax.random.split(key)
        repertoire, emitter_state, metrics = map_elites.update(
            repertoire, emitter_state, subkey
        )
        return (repertoire, emitter_state, key), metrics

    # Run the algorithm
    (
        repertoire,
        emitter_state,
        key,
    ), metrics = jax.lax.scan(
        update_scan_fn,
        (repertoire, emitter_state, key),
        (),
        length=num_iterations,
    )

    pytest.assume(repertoire is not None)
