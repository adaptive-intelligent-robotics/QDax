from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import brax

from brax.envs.env import State as EnvState
from qdax.types import (
    Action,
    Descriptors,
    Done,
    Fitness,
    Observation,
    Params,
    Reward,
    RNGKey,
    StateDescriptor,
)

@staticmethod
def training_inference(policy_model, params, obs):
    return policy_model.apply(params, obs)

@staticmethod
def get_deterministic_actions(parameters):
    loc, scale = jnp.split(parameters, 2, axis=-1)
    act = jnp.tanh(loc)
    return act


@flax.struct.dataclass
class Transition:
    """Stores data corresponding to a transition in an env."""

    obs: Observation
    next_obs: Observation
    rewards: Reward
    dones: Done
    actions: Action
    state_desc: StateDescriptor

def play_step(
    env_state: EnvState,
    policy_model,
    policy_params: Params,
    random_key: RNGKey,
    env: brax.envs.Env,
) -> Tuple[EnvState, Params, RNGKey, Transition]:
    """Play an environment step and return the updated state and the transition."""

    logits = training_inference(policy_model, policy_params, env_state.obs)
    # actions = parametric_action_distribution.sample(logits, key_sample)
    actions = get_deterministic_actions(logits)
    #actions = policy_model.apply(policy_params, env_state.obs)

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
    """Generates an episode according to the agent's policy, returns the final state of the
    episode and the transitions of the episode.
    """

    def _scannable_play_step_fn(
        carry: Tuple[EnvState, Params, RNGKey], unused_arg: Any
    ) -> Tuple[Tuple[EnvState, Params, RNGKey,], Transition]:
        env_state, policy_params, random_key, transitions = play_step_fn(*carry)
        return (env_state, policy_params, random_key), transitions

    (state, _, _), transitions = jax.lax.scan(
        _scannable_play_step_fn,
        (init_state, policy_params, key),
        (),
        length=episode_length,
    )
    return state, transitions