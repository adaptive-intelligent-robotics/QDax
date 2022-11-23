import functools
from functools import partial
from typing import Callable, Optional, Tuple

import brax.envs
import flax.linen as nn
import jax
import jax.numpy as jnp

import qdax.environments
from qdax import environments
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.mdp_utils import generate_unroll
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.types import (
    Descriptor,
    EnvState,
    ExtraScores,
    Fitness,
    Genotype,
    Params,
    RNGKey,
)


def create_policy_network_play_step_fn(
    env: brax.envs.Env,
    policy_network: nn.Module,
) -> Callable[
    [EnvState, Params, RNGKey], Tuple[EnvState, Params, RNGKey, QDTransition]
]:
    """
    Creates a function that when called, plays a step of the environment.

    Args:
        env: The BRAX environment.
        policy_network:  The policy network structure used for creating and evaluating
            policy controllers.

    Returns:
        default_play_step_fn: A function that plays a step of the environment.
    """
    # Define the function to play a step with the policy in the environment
    def default_play_step_fn(
        env_state: EnvState,
        policy_params: Params,
        random_key: RNGKey,
    ) -> Tuple[EnvState, Params, RNGKey, QDTransition]:
        """
        Play an environment step and return the updated EnvState and the transition.

        Args: env_state: The state of the environment (containing for instance the
        actor joint positions and velocities, the reward...). policy_params: The
        parameters of policies/controllers. random_key: JAX random key.

        Returns:
            next_state: The updated environment state.
            policy_params: The parameters of policies/controllers (unchanged).
            random_key: The updated random key.
            transition: containing some information about the transition: observation,
                reward, next observation, policy action...
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

    return default_play_step_fn


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
    play_step_fn: Callable[
        [EnvState, Params, RNGKey], Tuple[EnvState, Params, RNGKey, QDTransition]
    ],
    behavior_descriptor_extractor: Callable[[QDTransition, jnp.ndarray], Descriptor],
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """Evaluates policies contained in policies_params in parallel in
    deterministic or pseudo-deterministic environments.

    This rollout is only deterministic when all the init states are the same.
    If the init states are fixed but different, as a policy is not necessarily
    evaluated with the same environment everytime, this won't be determinist.
    When the init states are different, this is not purely stochastic.

    Args:
        policies_params: The parameters of closed-loop controllers/policies to evaluate.
        random_key: A jax random key
        episode_length: The maximal rollout length.
        play_step_fn: The function to play a step of the environment.
        behavior_descriptor_extractor: The function to extract the behavior descriptor.

    Returns:
        fitness: Array of fitnesses of all evaluated policies
        descriptor: Behavioural descriptors of all evaluated policies
        extra_scores: Additional information resulting from evaluation
        random_key: The updated random key.
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
    play_step_fn: Callable[
        [EnvState, Params, RNGKey], Tuple[EnvState, Params, RNGKey, QDTransition]
    ],
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

    Args:
        policies_params: The parameters of closed-loop controllers/policies to evaluate.
        random_key: A jax random key
        episode_length: The maximal rollout length.
        play_reset_fn: The function to reset the environment and obtain initial states.
        play_step_fn: The function to play a step of the environment.
        behavior_descriptor_extractor: The function to extract the behavior descriptor.

    Returns:
        fitness: Array of fitnesses of all evaluated policies
        descriptor: Behavioural descriptors of all evaluated policies
        extra_scores: Additional information resulting from the evaluation
        random_key: The updated random key.
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


def create_brax_scoring_fn(
    env: brax.envs.Env,
    policy_network: nn.Module,
    bd_extraction_fn: Callable[[QDTransition, jnp.ndarray], Descriptor],
    random_key: RNGKey,
    play_step_fn: Optional[
        Callable[
            [EnvState, Params, RNGKey], Tuple[EnvState, Params, RNGKey, QDTransition]
        ]
    ] = None,
    episode_length: int = 100,
    deterministic: bool = True,
    play_reset_fn: Optional[Callable[[RNGKey], EnvState]] = None,
) -> Tuple[
    Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]],
    RNGKey,
]:
    """
    Creates a scoring function to evaluate a policy in a BRAX task.

    Args:
        env: The BRAX environment.
        policy_network: The policy network controller.
        bd_extraction_fn: The behaviour descriptor extraction function.
        random_key: a random key used for stochastic operations.
        play_step_fn: the function used to perform environment rollouts and collect
            evaluation episodes. If None, we use create_policy_network_play_step_fn
            to generate it.
        episode_length: The maximal episode length.
        deterministic: Whether we reset the initial state of the robot to the same
            deterministic init_state or if we use the reset() function of the env.
        play_reset_fn: the function used to reset the environment to an initial state.
            Only used if deterministic is False. If None, we take env.reset as
            default reset function.

    Returns:
        The scoring function: a function that takes a batch of genotypes and compute
            their fitnesses and descriptors
        The updated random key.
    """
    if play_step_fn is None:
        play_step_fn = create_policy_network_play_step_fn(env, policy_network)

    # Deterministic case
    if deterministic:
        # Create the initial environment states
        random_key, subkey = jax.random.split(random_key)
        init_state = env.reset(subkey)

        # Define the function to deterministically reset the environment
        def deterministic_reset(key: RNGKey, init_state: EnvState) -> EnvState:
            return init_state

        play_reset_fn = partial(deterministic_reset, init_state=init_state)

    # Stochastic case
    elif play_reset_fn is None:
        play_reset_fn = env.reset

    scoring_fn = functools.partial(
        reset_based_scoring_function_brax_envs,
        episode_length=episode_length,
        play_reset_fn=play_reset_fn,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    return scoring_fn, random_key


def create_default_brax_task_components(
    env_name: str,
    random_key: RNGKey,
    episode_length: int = 100,
    mlp_policy_hidden_layer_sizes: Tuple[int, ...] = (64, 64),
    deterministic: bool = True,
) -> Tuple[
    brax.envs.Env,
    MLP,
    Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]],
    RNGKey,
]:
    """
    Creates default environment, policy network and scoring function for a BRAX task.

    Args:
        env_name: Name of the BRAX environment (e.g. "ant_omni", "walker2d_uni"...).
        random_key: Jax random key
        episode_length: The maximal rollout length.
        mlp_policy_hidden_layer_sizes: Hidden layer sizes of the policy network.
        deterministic: Whether we reset the initial state of the robot to the same
            deterministic init_state or if we use the reset() function of the env.

    Returns:
        env: The BRAX environment.
        policy_network: The policy network structure used for creating and evaluating
            policy controllers.
        scoring_fn: a function that takes a batch of genotypes and compute
            their fitnesses and descriptors.
        random_key: The updated random key.
    """
    env = environments.create(env_name, episode_length=episode_length)

    # Init policy network
    policy_layer_sizes = mlp_policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    bd_extraction_fn = qdax.environments.behavior_descriptor_extractor[env_name]

    scoring_fn, random_key = create_brax_scoring_fn(
        env,
        policy_network,
        bd_extraction_fn,
        random_key,
        episode_length=episode_length,
        deterministic=deterministic,
    )

    return env, policy_network, scoring_fn, random_key
