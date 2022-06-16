"""Functions similar to the ones in mdp_utils, the main difference is the assumption
that the policy parameters are included in the training state. By passing this whole
training state we can update running statistics for normalization for instance.

We are currently thinking about elegant ways to unify both in order to avoid code
repetition.
"""
# TODO: Uniformize with the functions in mdp_utils
from functools import partial
from typing import Any, Callable, Tuple

import jax
from brax.envs import State as EnvState

from qdax.core.neuroevolution.buffers.buffer import ReplayBuffer, Transition
from qdax.core.neuroevolution.mdp_utils import TrainingState
from qdax.types import Metrics


@partial(
    jax.jit,
    static_argnames=(
        "num_warmstart_steps",
        "play_step_fn",
        "env_batch_size",
    ),
)
def warmstart_buffer(
    replay_buffer: ReplayBuffer,
    training_state: TrainingState,
    env_state: EnvState,
    play_step_fn: Callable[
        [EnvState, TrainingState],
        Tuple[
            EnvState,
            TrainingState,
            Transition,
        ],
    ],
    num_warmstart_steps: int,
    env_batch_size: int,
) -> Tuple[ReplayBuffer, EnvState, TrainingState]:
    """Pre-populates the buffer with transitions. Returns the warmstarted buffer
    and the new state of the environment.
    """

    def _scan_play_step_fn(
        carry: Tuple[EnvState, TrainingState], unused_arg: Any
    ) -> Tuple[Tuple[EnvState, TrainingState], Transition]:
        env_state, training_state, transitions = play_step_fn(*carry)
        return (env_state, training_state), transitions

    (env_state, training_state), transitions = jax.lax.scan(
        _scan_play_step_fn,
        (env_state, training_state),
        (),
        length=num_warmstart_steps // env_batch_size,
    )
    replay_buffer = replay_buffer.insert(transitions)

    return replay_buffer, env_state, training_state


@partial(jax.jit, static_argnames=("play_step_fn", "episode_length"))
def generate_unroll(
    init_state: EnvState,
    training_state: TrainingState,
    episode_length: int,
    play_step_fn: Callable[
        [EnvState, TrainingState],
        Tuple[
            EnvState,
            TrainingState,
            Transition,
        ],
    ],
) -> Tuple[EnvState, TrainingState, Transition]:
    """Generates an episode according to the agent's policy, returns the final state of the
    episode and the transitions of the episode.
    """

    def _scan_play_step_fn(
        carry: Tuple[EnvState, TrainingState], unused_arg: Any
    ) -> Tuple[Tuple[EnvState, TrainingState], Transition]:
        env_state, training_state, transitions = play_step_fn(*carry)
        return (env_state, training_state), transitions

    (state, training_state), transitions = jax.lax.scan(
        _scan_play_step_fn,
        (init_state, training_state),
        (),
        length=episode_length,
    )
    return state, training_state, transitions


@partial(
    jax.jit,
    static_argnames=(
        "env_batch_size",
        "grad_updates_per_step",
        "play_step_fn",
        "update_fn",
    ),
)
def do_iteration_fn(
    training_state: TrainingState,
    env_state: EnvState,
    replay_buffer: ReplayBuffer,
    env_batch_size: int,
    grad_updates_per_step: float,
    play_step_fn: Callable[
        [EnvState, TrainingState],
        Tuple[
            EnvState,
            TrainingState,
            Transition,
        ],
    ],
    update_fn: Callable[
        [TrainingState, ReplayBuffer],
        Tuple[
            TrainingState,
            ReplayBuffer,
            Metrics,
        ],
    ],
) -> Tuple[TrainingState, EnvState, ReplayBuffer, Metrics]:
    """Performs one environment step (over all env simultaneously) followed by one
    training step. The number of updates is controlled by the parameter
    `grad_updates_per_step` (0 means no update while 1 means `env_batch_size`
    updates). Returns the updated states, the updated buffer and the aggregated
    metrics.
    """

    def _scan_update_fn(
        carry: Tuple[TrainingState, ReplayBuffer], unused_arg: Any
    ) -> Tuple[Tuple[TrainingState, ReplayBuffer], Metrics]:
        training_state, replay_buffer, metrics = update_fn(*carry)
        return (training_state, replay_buffer), metrics

    # play steps in the environment
    env_state, training_state, transitions = play_step_fn(env_state, training_state)

    # insert transitions in replay buffer
    replay_buffer = replay_buffer.insert(transitions)
    num_updates = int(grad_updates_per_step * env_batch_size)

    (training_state, replay_buffer), metrics = jax.lax.scan(
        _scan_update_fn,
        (training_state, replay_buffer),
        (),
        length=num_updates,
    )

    return training_state, env_state, replay_buffer, metrics
