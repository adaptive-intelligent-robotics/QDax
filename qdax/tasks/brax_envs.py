# TODO: Show this to Bryan
import functools
from functools import partial
from typing import Any, Callable, Tuple

import brax.envs
import flax.linen as nn
import jax
import jax.numpy as jnp
from typing_extensions import TypeAlias

import qdax.environments
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.mdp_utils import generate_unroll
from qdax.types import (
    Descriptor,
    EnvState,
    ExtraScores,
    Fitness,
    Genotype,
    Params,
    RNGKey,
)

PlayStepFnType: TypeAlias = Callable[
    [EnvState, Params, RNGKey], Tuple[EnvState, Params, RNGKey, QDTransition]
]


def create_policy_network_play_step_fn(
    env: brax.envs.Env,
    policy_network: nn.Module,
) -> PlayStepFnType:
    # Define the function to play a step with the policy in the environment
    def play_step_fn(
        env_state: EnvState,
        policy_params: Params,
        random_key: RNGKey,
    ) -> Tuple[EnvState, Params, RNGKey, QDTransition]:
        """
        Play an environment step and return the updated EnvState and the transition.
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

    return play_step_fn


def create_brax_scoring_function_fn(
    env_name: str,
    policy_network: nn.Module,
    batch_size: int,
    random_key: RNGKey,
    episode_length: int = 100,
    **kwargs_env_def: Any,
) -> Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]]:
    """Returns a function that when called, creates a scoring function.
    Please use namespace to avoid confusion between this function and
    brax.envs.get_scoring_function_fn.
    """
    env = qdax.environments.create(
        env_name, episode_length=episode_length, **kwargs_env_def
    )
    play_step_fn = create_policy_network_play_step_fn(env, policy_network)

    bd_extraction_fn = qdax.environments.behavior_descriptor_extractor[env_name]

    # Create the initial environment states
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=batch_size, axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)

    scoring_fn = functools.partial(
        scoring_function_brax_envs,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    return scoring_fn


@partial(
    jax.jit,
    static_argnames=(
        "episode_length",
        "play_reset_fn",
        "play_step_fn",
        "behavior_descriptor_extractor",
    ),
)
def reset_based_scoring_function_brax_envs(
    policies_params: Genotype,
    random_key: RNGKey,
    episode_length: int,
    play_reset_fn: Callable[[RNGKey], EnvState],
    play_step_fn: PlayStepFnType,
    behavior_descriptor_extractor: Callable[[QDTransition, jnp.ndarray], Descriptor],
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """Evaluates policies contained in policies_params in parallel.
    The play_reset_fn function allows for a more general scoring_function that can be
    called with different batch-size and not only with a batch-size of the same
    dimension as init_states.

    To define purely stochastic environments, using the reset function from the
    environment, use "play_reset_fn = env.reset".

    To define purely deterministic environments, as in "scoring_function", generate
    a single init_state using "init_state = env.reset(random_key)", then use
    "play_reset_fn = lambda random_key: init_state".
    """

    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, jax.tree_leaves(policies_params)[0].shape[0])
    reset_fn = jax.vmap(play_reset_fn)
    init_states = reset_fn(keys)

    fitnesses, descriptors, extra_scores, random_key = scoring_function_brax_envs(
        policies_params=policies_params,
        random_key=random_key,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=behavior_descriptor_extractor,
    )

    return fitnesses, descriptors, extra_scores, random_key


@partial(
    jax.jit,
    static_argnames=(
        "episode_length",
        "play_step_fn",
        "behavior_descriptor_extractor",
    ),
)
def scoring_function_brax_envs(
    policies_params: Genotype,
    random_key: RNGKey,
    init_states: EnvState,
    episode_length: int,
    play_step_fn: PlayStepFnType,
    behavior_descriptor_extractor: Callable[[QDTransition, jnp.ndarray], Descriptor],
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """Evaluates policies contained in policies_params in parallel in
    deterministic or pseudo-deterministic environments.

    This rollout is only deterministic when all the init states are the same.
    If the init states are fixed but different, as a policy is not necessarly
    evaluated with the same environment everytime, this won't be determinist.
    When the init states are different, this is not purely stochastic.
    """

    # Perform rollouts with each policy
    random_key, subkey = jax.random.split(random_key)
    unroll_fn = partial(
        generate_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        random_key=subkey,
    )

    _final_state, data = jax.vmap(unroll_fn)(init_states, policies_params)

    # create a mask to extract data properly
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=1), 0, 1)
    mask = jnp.roll(is_done, 1, axis=1)
    mask = mask.at[:, 0].set(0)

    # Scores - add offset to ensure positive fitness (through positive rewards)
    fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1)
    descriptors = behavior_descriptor_extractor(data, mask)

    return (
        fitnesses,
        descriptors,
        {
            "transitions": data,
        },
        random_key,
    )
