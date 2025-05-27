from functools import partial
from typing import Any, Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jumanji
from chex import ArrayTree
from typing_extensions import TypeAlias

from qdax.core.neuroevolution.buffers.buffer import QDTransition, Transition
from qdax.custom_types import (
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Observation,
    Params,
    RNGKey,
)

JumanjiState: TypeAlias = ArrayTree
JumanjiTimeStep: TypeAlias = jumanji.types.TimeStep


def make_policy_network_play_step_fn_jumanji(
    env: jumanji.env.Environment,
    policy_network: nn.Module,
    observation_processing: Callable[[jumanji.types.Observation], Observation],
) -> Callable[
    [JumanjiState, JumanjiTimeStep, Params, RNGKey],
    Tuple[JumanjiState, JumanjiTimeStep, Params, RNGKey, QDTransition],
]:
    """
    Creates a function that when called, plays a step of the environment.

    Args:
        env: The jumanji environment.
        policy_network:  The policy network structure used for creating and evaluating
            policy controllers.
        observation_processing: a method to do modify the observation from the
        environment.

    Returns:
        default_play_step_fn: A function that plays a step of the environment.
    """

    # Define the function to play a step with the policy in the environment
    def default_play_step_fn(
        env_state: JumanjiState,
        timestep: JumanjiTimeStep,
        policy_params: Params,
        key: RNGKey,
    ) -> Tuple[JumanjiState, JumanjiTimeStep, Params, RNGKey, QDTransition]:
        """Play an environment step and return the updated state and the transition.
        Everything is deterministic in this simple example.
        """

        network_input = observation_processing(timestep.observation)

        proba_action = policy_network.apply(policy_params, network_input)

        action = jnp.argmax(proba_action)

        state_desc = None
        next_state, next_timestep = env.step(env_state, action)

        next_state_desc = None

        transition = QDTransition(
            obs=timestep.observation,
            next_obs=next_timestep.observation,
            rewards=next_timestep.reward,
            dones=jnp.where(next_timestep.last(), jnp.array(1), jnp.array(0)),
            actions=action,
            truncations=jnp.array(0),
            state_desc=state_desc,
            next_state_desc=next_state_desc,
        )

        return next_state, next_timestep, policy_params, key, transition

    return default_play_step_fn


def generate_jumanji_unroll(
    init_state: JumanjiState,
    init_timestep: JumanjiTimeStep,
    policy_params: Params,
    key: RNGKey,
    episode_length: int,
    play_step_fn: Callable[
        [JumanjiState, JumanjiTimeStep, Params, RNGKey],
        Tuple[
            JumanjiState,
            JumanjiTimeStep,
            Params,
            RNGKey,
            Transition,
        ],
    ],
) -> Tuple[JumanjiState, JumanjiTimeStep, Transition]:
    """Generates an episode according to the agent's policy, returns the final state of
    the episode and the transitions of the episode.

    Args:
        init_state: first state of the rollout.
        policy_params: params of the individual.
        key: random key for stochasiticity handling.
        episode_length: length of the rollout.
        play_step_fn: function describing how a step need to be taken.

    Returns:
        A new state, the experienced transition.
    """

    def _scan_play_step_fn(
        carry: Tuple[JumanjiState, JumanjiTimeStep, Params, RNGKey], unused_arg: Any
    ) -> Tuple[Tuple[JumanjiState, JumanjiTimeStep, Params, RNGKey], Transition]:
        env_state, timestep, policy_params, key, transitions = play_step_fn(*carry)
        return (env_state, timestep, policy_params, key), transitions

    (state, timestep, _, _), transitions = jax.lax.scan(
        _scan_play_step_fn,
        (init_state, init_timestep, policy_params, key),
        (),
        length=episode_length,
    )
    return state, timestep, transitions


@partial(
    jax.jit,
    static_argnames=(
        "episode_length",
        "play_step_fn",
        "descriptor_extractor",
    ),
)
def jumanji_scoring_function(
    policies_params: Genotype,
    key: RNGKey,
    init_states: JumanjiState,
    init_timesteps: JumanjiTimeStep,
    episode_length: int,
    play_step_fn: Callable[
        [JumanjiState, JumanjiTimeStep, Params, RNGKey],
        Tuple[JumanjiState, JumanjiTimeStep, Params, RNGKey, QDTransition],
    ],
    descriptor_extractor: Callable[[QDTransition, jnp.ndarray], Descriptor],
) -> Tuple[Fitness, Descriptor, ExtraScores]:
    """Evaluates policies contained in policies_params in parallel in
    deterministic or pseudo-deterministic environments.

    This rollout is only deterministic when all the init states are the same.
    If the init states are fixed but different, as a policy is not necessary
    evaluated with the same environment every time, this won't be deterministic.
    When the init states are different, this is not purely stochastic.
    """

    # Step environments
    unroll_fn = partial(
        generate_jumanji_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
    )
    keys = jax.random.split(key, jax.tree.leaves(policies_params)[0].shape[0])
    _, _, data = jax.vmap(unroll_fn)(init_states, init_timesteps, policies_params, keys)

    # Create a mask to extract data properly
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=1), 0, 1)
    mask = jnp.roll(is_done, 1, axis=1)
    mask = mask.at[:, 0].set(0)

    # Evaluate
    fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1)
    descriptors = descriptor_extractor(data, mask)

    return fitnesses, descriptors, {"transitions": data}
