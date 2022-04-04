from functools import partial
from typing import Any, Callable, Tuple

import flax
import jax
from brax.envs import State as EnvState

from qdax.types import (
    Action,
    Done,
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
