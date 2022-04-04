from functools import partial
from typing import Any, Callable, Tuple

import brax
import flax
import jax
import jax.numpy as jnp
from brax.envs import State as EnvState

from qdax.types import (
    Action,
    Descriptor,
    Done,
    Fitness,
    Genotype,
    Observation,
    Params,
    Reward,
    RNGKey,
    StateDescriptor,
)


@flax.struct.dataclass
class Transition:
    """Stores data corresponding to a transition in an env."""

    obs: Observation
    next_obs: Observation
    rewards: Reward
    dones: Done
    actions: Action
    state_desc: StateDescriptor


@partial(jax.jit, static_argnames=("play_step_fn", "episode_length"))
def generate_unroll(
    init_state: EnvState,
    policy_params: Params,
    key: RNGKey,
    episode_length: int,
    play_step_fn: Callable[
        [EnvState, Params, RNGKey],
        Tuple[
            EnvState,
            Params,
            RNGKey,
            Transition,
        ],
    ],
) -> Tuple[EnvState, Transition]:
    """
    Generates an episode according to the agent's policy, returns the final state
    of the episode and the transitions of the episode.
    """

    def _scan_play_step_fn(
        carry: Tuple[EnvState, Params, RNGKey], unused_arg: Any
    ) -> Tuple[Tuple[EnvState, Params, RNGKey], Transition]:
        env_state, policy_params, random_key, transitions = play_step_fn(*carry)
        return (env_state, policy_params, random_key), transitions

    (state, _, _), transitions = jax.lax.scan(
        _scan_play_step_fn,
        (init_state, policy_params, key),
        (),
        length=episode_length,
    )
    return state, transitions


@partial(jax.jit, static_argnames=("env", "policy_network"))
def play_step(
    env_state: EnvState,
    policy_params: Params,
    random_key: RNGKey,
    env: brax.envs.Env,
    policy_network: flax.linen.Module
) -> Tuple[EnvState, Params, RNGKey, Transition]:
    """Play an environment step and return the updated state and the transition."""

    actions = policy_network.apply(policy_params, env_state.obs)
    next_state = env.step(env_state, actions)

    transition = Transition(
        obs=env_state.obs,
        next_obs=next_state.obs,
        rewards=next_state.reward,
        dones=next_state.done,
        actions=actions,
        state_desc=env_state.info["state_descriptor"],
    )

    return next_state, policy_params, random_key, transition


@partial(
    jax.jit,
    static_argnames=(
        "episode_length",
        "env",
        "behavior_descriptor_extractor",
    ),
)
def scoring_function(
    policies_params: Genotype,
    init_states: brax.envs.State,
    episode_length: int,
    random_key: RNGKey,
    env: brax.envs.Env,
    policy_network: flax.linen.Module,
    behavior_descriptor_extractor: Callable[[Transition, jnp.ndarray], Descriptor],
) -> Tuple[Fitness, Descriptor]:
    """Evaluate policies contained in flatten_variables in parallel

    This rollout is only determinist when all the init states are the same.
    If the init states are fixed but different, as a policy is not necessarily
    evaluated with the same environment everytime, this won't be determinist.

    When the init states, this is not purely stochastic. This choice was made
    for performance reason, as the reset function of brax envs is quite
    time-consuming. If pure stochasticity is needed for a use case, please open
    an issue.
    """
    play_step_fn = partial(play_step, env=env, policy_network=policy_network)

    # Perform rollouts with each policy
    unroll_fn = partial(
        generate_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        key=random_key,
    )

    _final_state, data = jax.vmap(unroll_fn)(init_states, policies_params)

    # Create a mask to extract data properly
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=1), 0, 1)
    mask = jnp.roll(is_done, 1, axis=1)
    mask = mask.at[:, 0].set(0)

    # Scores - add offset to ensure positive fitness (through positive rewards)
    fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1)
    descriptors = behavior_descriptor_extractor(data, mask)

    return fitnesses, descriptors
