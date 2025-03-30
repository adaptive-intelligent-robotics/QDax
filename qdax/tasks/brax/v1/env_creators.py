import functools
from functools import partial
from typing import Callable, Optional, Tuple

import brax.envs
import flax.linen as nn
import jax
import jax.numpy as jnp

import qdax.tasks.brax.v1 as environments
from qdax.core.neuroevolution.buffers.buffer import QDTransition, Transition
from qdax.core.neuroevolution.mdp_utils import generate_unroll
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import (
    Descriptor,
    EnvState,
    ExtraScores,
    Fitness,
    Genotype,
    Observation,
    Params,
    RNGKey,
)


def make_policy_network_play_step_fn_brax(
    env: brax.envs.Env,
    policy_network: nn.Module,
) -> Callable[
    [EnvState, Params, RNGKey], Tuple[EnvState, Params, RNGKey, QDTransition]
]:
    """
    Creates a function that when called, plays a step of the environment.

    Args:
        env: The Brax environment.
        policy_network: The policy network structure used for creating and evaluating
            policy controllers.

    Returns:
        default_play_step_fn: A function that plays a step of the environment.
    """

    # Define the function to play a step with the policy in the environment
    def default_play_step_fn(
        env_state: EnvState,
        policy_params: Params,
        key: RNGKey,
    ) -> Tuple[EnvState, Params, RNGKey, QDTransition]:
        """
        Play an environment step and return the updated EnvState and the transition.

        Args:
            env_state: The state of the environment (containing for instance the
                actor joint positions and velocities, the reward...).
            policy_params: The parameters of policies/controllers. key: JAX random key.

        Returns:
            next_env_state: The updated environment state.
            policy_params: The parameters of policies/controllers (unchanged).
            key: The updated random key.
            transition: containing some information about the transition: observation,
                reward, next observation, policy action...
        """

        actions = policy_network.apply(policy_params, env_state.obs)

        state_desc = env_state.info["state_descriptor"]
        next_env_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_env_state.obs,
            rewards=next_env_state.reward,
            dones=next_env_state.done,
            actions=actions,
            truncations=next_env_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_env_state.info["state_descriptor"],
        )

        return next_env_state, policy_params, key, transition

    return default_play_step_fn


def get_mask_from_transitions(
    data: Transition,
) -> jnp.ndarray:
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=1), 0, 1)
    mask = jnp.roll(is_done, 1, axis=1)
    mask = mask.at[:, 0].set(0)
    return mask


@partial(
    jax.jit,
    static_argnames=(
        "episode_length",
        "play_reset_fn",
        "play_step_fn",
        "descriptor_extractor",
    ),
)
def scoring_function_brax_envs(
    policies_params: Genotype,
    key: RNGKey,
    episode_length: int,
    play_reset_fn: Callable[[RNGKey], EnvState],
    play_step_fn: Callable[
        [EnvState, Params, RNGKey], Tuple[EnvState, Params, RNGKey, QDTransition]
    ],
    descriptor_extractor: Callable[[QDTransition, jnp.ndarray], Descriptor],
) -> Tuple[Fitness, Descriptor, ExtraScores]:
    """Evaluates policies contained in policies_params in parallel.
    The play_reset_fn function allows for a more general scoring_function that can be
    called with different batch-size and not only with a batch-size of the same
    dimension as init_states.

    To define purely stochastic environments, using the reset function from the
    environment, use "play_reset_fn = env.reset".

    To define purely deterministic environments, as in "scoring_function", generate
    a single init_state using "init_state = env.reset(key)", then use
    "play_reset_fn = lambda key: init_state".

    Args:
        policies_params: The parameters of closed-loop controllers/policies to evaluate.
        key: A jax random key
        episode_length: The maximal rollout length.
        play_reset_fn: The function to reset the environment and obtain initial states.
        play_step_fn: The function to play a step of the environment.
        descriptor_extractor: The function to extract the descriptor.

    Returns:
        fitness: Array of fitnesses of all evaluated policies
        descriptor: Behavioural descriptors of all evaluated policies
        extra_scores: Additional information resulting from the evaluation
    """

    # Reset environments
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, jax.tree.leaves(policies_params)[0].shape[0])
    init_states = jax.vmap(play_reset_fn)(keys)

    # Step environments
    unroll_fn = partial(
        generate_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
    )
    keys = jax.random.split(key, jax.tree.leaves(policies_params)[0].shape[0])
    _, data = jax.vmap(unroll_fn)(init_states, policies_params, keys)

    # Create a mask to extract data properly
    mask = get_mask_from_transitions(data)

    # Evaluate
    fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1)
    descriptors = descriptor_extractor(data, mask)

    return fitnesses, descriptors, {"transitions": data}


def create_brax_scoring_fn(
    env: brax.envs.Env,
    policy_network: nn.Module,
    descriptor_extraction_fn: Callable[[QDTransition, jnp.ndarray], Descriptor],
    key: RNGKey,
    play_step_fn: Optional[
        Callable[
            [EnvState, Params, RNGKey], Tuple[EnvState, Params, RNGKey, QDTransition]
        ]
    ] = None,
    episode_length: int = 100,
    deterministic: bool = True,
    play_reset_fn: Optional[Callable[[RNGKey], EnvState]] = None,
) -> Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores]]:
    """
    Creates a scoring function to evaluate a policy in a BRAX task.

    Args:
        env: The Brax environment.
        policy_network: The policy network controller.
        descriptor_extraction_fn: The behaviour descriptor extraction function.
        key: a random key used for stochastic operations.
        play_step_fn: the function used to perform environment rollouts and collect
            evaluation episodes. If None, we use make_policy_network_play_step_fn_brax
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
    """
    if play_step_fn is None:
        play_step_fn = make_policy_network_play_step_fn_brax(env, policy_network)

    # Deterministic case
    if deterministic:
        # Create the initial environment states
        key, subkey = jax.random.split(key)
        init_state = env.reset(subkey)

        # Define the function to deterministically reset the environment
        def deterministic_reset(_: RNGKey, _init_state: EnvState) -> EnvState:
            return _init_state

        play_reset_fn = partial(deterministic_reset, _init_state=init_state)

    # Stochastic case
    elif play_reset_fn is None:
        play_reset_fn = env.reset

    scoring_fn = functools.partial(
        scoring_function_brax_envs,
        episode_length=episode_length,
        play_reset_fn=play_reset_fn,
        play_step_fn=play_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
    )

    return scoring_fn


def create_default_brax_task_components(
    env_name: str,
    key: RNGKey,
    episode_length: int = 100,
    mlp_policy_hidden_layer_sizes: Tuple[int, ...] = (64, 64),
    deterministic: bool = True,
) -> Tuple[
    brax.envs.Env,
    MLP,
    Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores]],
]:
    """
    Creates default environment, policy network and scoring function for a BRAX task.

    Args:
        env_name: Name of the BRAX environment (e.g. "ant_omni", "walker2d_uni"...).
        key: JAX random key
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
    """
    env = environments.create(env_name, episode_length=episode_length)

    # Init policy network
    policy_layer_sizes = mlp_policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    descriptor_extraction_fn = environments.descriptor_extractor[env_name]

    scoring_fn = create_brax_scoring_fn(
        env,
        policy_network,
        descriptor_extraction_fn,
        key,
        episode_length=episode_length,
        deterministic=deterministic,
    )

    return env, policy_network, scoring_fn


def get_aurora_scoring_fn(
    scoring_fn: Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores]],
    observation_extractor_fn: Callable[[Transition], Observation],
    observations_key: str,
) -> Callable[[Genotype, RNGKey], Tuple[Fitness, Optional[Descriptor], ExtraScores]]:
    """Evaluates policies contained in flatten_variables in parallel

    This rollout is only deterministic when all the init states are the same.
    If the init states are fixed but different, as a policy is not necessary
    evaluated with the same environment every time, this won't be deterministic.

    When the init states are different, this is not purely stochastic. This
    choice was made for performance reason, as the reset function of brax envs
    is quite time-consuming. If pure stochasticity of the environment is needed
    for a use case, please open an issue.
    """

    @functools.wraps(scoring_fn)
    def _wrapper(
        params: Params, key: RNGKey  # Perform rollouts with each policy
    ) -> Tuple[Fitness, Optional[Descriptor], ExtraScores]:
        fitnesses, _, extra_scores = scoring_fn(params, key)
        data = extra_scores["transitions"]

        observation = observation_extractor_fn(data)  # type: ignore
        extra_scores[observations_key] = observation

        return fitnesses, None, extra_scores

    return _wrapper
