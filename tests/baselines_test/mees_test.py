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
from qdax.tasks.brax_envs import scoring_function_brax_envs


def test_mees() -> None:
    env_name = "walker2d_uni"
    episode_length = 100
    num_iterations = 10
    seed = 42
    policy_hidden_layer_sizes = (64, 64)
    num_init_cvt_samples = 1000
    num_centroids = 50
    min_bd = 0.0
    max_bd = 1.0

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
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=sample_number, axis=0)
    init_states_samples = reset_fn(keys)
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=1, axis=0)
    init_states = reset_fn(keys)

    # Prepare the scoring function for samples and final indivs
    bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function_brax_envs,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )
    scoring_samples_fn = functools.partial(
        scoring_function_brax_envs,
        init_states=init_states_samples,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
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
        scoring_fn=scoring_samples_fn,
        num_descriptors=env.behavior_descriptor_length,
    )

    # Compute the centroids
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_bd,
        maxval=max_bd,
        random_key=random_key,
    )

    # Instantiate MAP Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=mees_emitter,
        metrics_function=metrics_function,
    )

    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )

    @jax.jit
    def update_scan_fn(carry: Any, unused: Any) -> Any:
        # iterate over grid
        repertoire, emitter_state, metrics, random_key = map_elites.update(*carry)

        return (repertoire, emitter_state, random_key), metrics

    # Run the algorithm
    (
        repertoire,
        emitter_state,
        random_key,
    ), metrics = jax.lax.scan(
        update_scan_fn,
        (repertoire, emitter_state, random_key),
        (),
        length=num_iterations,
    )

    pytest.assume(repertoire is not None)
