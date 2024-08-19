import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import pytest

from qdax import environments
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.arm import arm_scoring_function
from qdax.tasks.brax_envs import scoring_function_brax_envs
from qdax.types import EnvState, Params, RNGKey
from qdax.utils.uncertainty_metrics import (
    reevaluation_function,
    reevaluation_reproducibility_function,
)


def test_uncertainty_metrics() -> None:
    seed = 42
    num_reevals = 512
    batch_size = 512
    num_init_cvt_samples = 50000
    num_centroids = 1024

    # Init a random key
    random_key = jax.random.PRNGKey(seed)

    # First, init a deterministic environment
    scoring_fn = arm_scoring_function

    # Init policies
    init_policies = jax.random.uniform(
        random_key, shape=(batch_size, 8), minval=0, maxval=1
    )

    # Evaluate in the deterministic environment
    fitnesses, descriptors, extra_scores, random_key = scoring_fn(
        init_policies, random_key
    )

    # Initialise a container
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=2,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=jnp.array([0.0, 0.0]),
        maxval=jnp.array([1.0, 1.0]),
        random_key=random_key,
    )
    repertoire = MapElitesRepertoire.init(
        genotypes=init_policies,
        fitnesses=fitnesses,
        descriptors=descriptors,
        centroids=centroids,
        extra_scores=extra_scores,
    )

    # Initialise an empty container for corrected repertoire
    fitnesses = jnp.full_like(fitnesses, -jnp.inf)
    empty_corrected_repertoire = MapElitesRepertoire.init(
        genotypes=init_policies,
        fitnesses=fitnesses,
        descriptors=descriptors,
        centroids=centroids,
        extra_scores=extra_scores,
    )

    # Test that reevaluation_function accurately predicts no change
    corrected_repertoire, random_key = reevaluation_function(
        repertoire=repertoire,
        empty_corrected_repertoire=empty_corrected_repertoire,
        scoring_fn=scoring_fn,
        num_reevals=num_reevals,
        random_key=random_key,
    )
    pytest.assume(
        jnp.allclose(
            corrected_repertoire.fitnesses, repertoire.fitnesses, rtol=1e-05, atol=1e-05
        )
    )

    # Test that reevaluation_reproducibility_function accurately predicts no change
    (
        corrected_repertoire,
        fit_reproducibility_repertoire,
        desc_reproducibility_repertoire,
        random_key,
    ) = reevaluation_reproducibility_function(
        repertoire=repertoire,
        empty_corrected_repertoire=empty_corrected_repertoire,
        scoring_fn=scoring_fn,
        num_reevals=num_reevals,
        random_key=random_key,
    )
    pytest.assume(
        jnp.allclose(
            corrected_repertoire.fitnesses, repertoire.fitnesses, rtol=1e-05, atol=1e-05
        )
    )
    zero_fitnesses = jnp.where(
        repertoire.fitnesses > -jnp.inf,
        0.0,
        -jnp.inf,
    )
    pytest.assume(
        jnp.allclose(
            fit_reproducibility_repertoire.fitnesses,
            zero_fitnesses,
            rtol=1e-05,
            atol=1e-05,
        )
    )
    pytest.assume(
        jnp.allclose(
            desc_reproducibility_repertoire.fitnesses,
            zero_fitnesses,
            rtol=1e-05,
            atol=1e-05,
        )
    )

    # Second, init a Brax environment
    env_name = "walker2d_uni"
    episode_length = 100
    policy_hidden_layer_sizes = (64, 64)
    env = environments.create(env_name, episode_length=episode_length)

    # Init policy network
    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=batch_size, axis=0)
    fake_batch = jnp.zeros(shape=(batch_size, env.observation_size))
    init_policies = jax.vmap(policy_network.init)(keys, fake_batch)

    # Define the fonction to play a step with the policy in the environment
    def play_step_fn(
        env_state: EnvState,
        policy_params: Params,
        random_key: RNGKey,
    ) -> Tuple[EnvState, Params, RNGKey, QDTransition]:

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
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=batch_size, axis=0)
    init_states = reset_fn(keys)

    # Create the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function_brax_envs,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Evaluate in the Brax environment
    fitnesses, descriptors, extra_scores, random_key = scoring_fn(
        init_policies, random_key
    )

    # Initialise a container
    min_bd, max_bd = env.behavior_descriptor_limits
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_bd,
        maxval=max_bd,
        random_key=random_key,
    )
    repertoire = MapElitesRepertoire.init(
        genotypes=init_policies,
        fitnesses=fitnesses,
        descriptors=descriptors,
        centroids=centroids,
        extra_scores=extra_scores,
    )

    # Initialise an empty container for corrected repertoire
    fitnesses = jnp.full_like(fitnesses, -jnp.inf)
    empty_corrected_repertoire = MapElitesRepertoire.init(
        genotypes=init_policies,
        fitnesses=fitnesses,
        descriptors=descriptors,
        centroids=centroids,
        extra_scores=extra_scores,
    )

    # Test that reevaluation_function runs and keeps at least one solution
    keys = jnp.repeat(
        jnp.expand_dims(subkey, axis=0), repeats=num_centroids, axis=0
    )
    init_states = reset_fn(keys)
    reeval_scoring_fn = functools.partial(
        scoring_function_brax_envs,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )
    corrected_repertoire, random_key = reevaluation_function(
        repertoire=repertoire,
        empty_corrected_repertoire=empty_corrected_repertoire,
        scoring_fn=reeval_scoring_fn,
        num_reevals=num_reevals,
        random_key=random_key,
    )
    pytest.assume(jnp.any(fit_reproducibility_repertoire.fitnesses > -jnp.inf))
