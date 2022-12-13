from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import jumanji
from typing_extensions import TypeAlias

from qdax.core.neuroevolution.buffers.buffer import QDTransition, Transition
from qdax.types import (
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    JumanjiState,
    Params,
    RNGKey,
)

TimeStep: TypeAlias = jumanji.types.TimeStep


@partial(jax.jit, static_argnames=("play_step_fn", "episode_length"))
def generate_jumanji_unroll(
    init_state: JumanjiState,
    init_timestep: TimeStep,
    policy_params: Params,
    random_key: RNGKey,
    episode_length: int,
    play_step_fn: Callable[
        [JumanjiState, TimeStep, Params, RNGKey],
        Tuple[
            JumanjiState,
            TimeStep,
            Params,
            RNGKey,
            Transition,
        ],
    ],
) -> Tuple[JumanjiState, TimeStep, Transition]:
    """Generates an episode according to the agent's policy, returns the final state of
    the episode and the transitions of the episode.

    Args:
        init_state: first state of the rollout.
        policy_params: params of the individual.
        random_key: random key for stochasiticity handling.
        episode_length: length of the rollout.
        play_step_fn: function describing how a step need to be taken.

    Returns:
        A new state, the experienced transition.
    """

    def _scan_play_step_fn(
        carry: Tuple[JumanjiState, TimeStep, Params, RNGKey], unused_arg: Any
    ) -> Tuple[Tuple[JumanjiState, TimeStep, Params, RNGKey], Transition]:
        env_state, timestep, policy_params, random_key, transitions = play_step_fn(
            *carry
        )
        return (env_state, timestep, policy_params, random_key), transitions

    (state, timestep, _, _), transitions = jax.lax.scan(
        _scan_play_step_fn,
        (init_state, init_timestep, policy_params, random_key),
        (),
        length=episode_length,
    )
    return state, timestep, transitions


@partial(
    jax.jit,
    static_argnames=(
        "episode_length",
        "play_step_fn",
        "behavior_descriptor_extractor",
    ),
)
def jumanji_scoring_function(
    policies_params: Genotype,
    random_key: RNGKey,
    init_states: JumanjiState,
    init_timesteps: TimeStep,
    episode_length: int,
    play_step_fn: Callable[
        [JumanjiState, TimeStep, Params, RNGKey, jumanji.env.Environment],
        Tuple[JumanjiState, TimeStep, Params, RNGKey, QDTransition],
    ],
    behavior_descriptor_extractor: Callable[[QDTransition, jnp.ndarray], Descriptor],
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """Evaluates policies contained in policies_params in parallel in
    deterministic or pseudo-deterministic environments.

    This rollout is only deterministic when all the init states are the same.
    If the init states are fixed but different, as a policy is not necessarly
    evaluated with the same environment everytime, this won't be determinist.
    When the init states are different, this is not purely stochastic.
    """

    print("Look a this policy params: ", policies_params)

    # Perform rollouts with each policy
    random_key, subkey = jax.random.split(random_key)
    unroll_fn = partial(
        generate_jumanji_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        random_key=subkey,
    )

    _final_state, _final_timestep, data = jax.vmap(unroll_fn)(
        init_states, init_timesteps, policies_params
    )

    # create a mask to extract data properly
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=1), 0, 1)
    mask = jnp.roll(is_done, 1, axis=1)
    mask = mask.at[:, 0].set(0)

    # # replace NaNs by zeros
    # corrected_rewards = jnp.nan_to_num(
    #     data.rewards, copy=True, nan=0.0, posinf=None, neginf=None
    # )

    # Scores - add offset to ensure positive fitness (through positive rewards)
    # fitnesses = jnp.sum(corrected_rewards * (1.0 - mask), axis=1)
    fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1)
    # descriptors = behavior_descriptor_extractor(data, mask)
    descriptors = jnp.array([0.0])

    print("Look at this fitness: ", fitnesses)
    print("Look at this descriptor: ", descriptors)

    print("Look at this transition dones: ", data.dones)
    print("Look at this transition rewards: ", data.rewards)

    return (
        fitnesses,
        descriptors,
        {
            "transitions": data,
        },
        random_key,
    )
