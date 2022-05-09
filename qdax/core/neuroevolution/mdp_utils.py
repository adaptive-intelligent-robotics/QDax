from functools import partial
from typing import Any, Callable, Dict, Tuple, Union

import brax
import jax
import jax.numpy as jnp
from brax.envs import State as EnvState
from qdax.core.neuroevolution.buffers.buffers import QDTransition, Transition
from qdax.types import Descriptor, Fitness, Genotype, Params, RNGKey


@partial(jax.jit, static_argnames=("play_step_fn", "episode_length"))
def generate_unroll(
    init_state: EnvState,
    policy_params: Params,
    random_key: RNGKey,
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
    """Generates an episode according to the agent's policy, returns the final state of
    the episode and the transitions of the episode.

    Args:
        init_state: first state of the rollout.
        policy_params: params of the individual.
        random_key: random key for stochasiticity handling.
        episode_length: length of the rollout.
        play_step_fn: function describing how a step need to be taken.

    Returns:
        A new state, the experienced transition, a new key.
    """
    def _scannable_play_step_fn(
        carry: Tuple[EnvState, Params, RNGKey], unused_arg: Any
    ) -> Tuple[Tuple[EnvState, Params, RNGKey], Transition]:
        env_state, policy_params, random_key, transitions = play_step_fn(*carry)
        return (env_state, policy_params, random_key), transitions

    (state, _, _), transitions = jax.lax.scan(
        _scannable_play_step_fn,
        (init_state, policy_params, random_key),
        (),
        length=episode_length,
    )
    return state, transitions


@partial(
    jax.jit,
    static_argnames=(
        "episode_length",
        "play_step_fn",
        "behavior_descriptor_extractor",
    ),
)
def scoring_function(
    policies_params: Genotype,
    init_states: brax.envs.State,
    episode_length: int,
    random_key: RNGKey,
    play_step_fn: Callable[
        [EnvState, Params, RNGKey, brax.envs.Env],
        Tuple[EnvState, Params, RNGKey, QDTransition],
    ],
    behavior_descriptor_extractor: Callable[[QDTransition, jnp.ndarray], Descriptor],
) -> Tuple[Fitness, Descriptor, Dict[str, Union[jnp.ndarray, QDTransition]], RNGKey]:
    """Evaluates policies contained in flatten_variables in parallel

    This rollout is only deterministic when all the init states are the same.
    If the init states are fixed but different, as a policy is not necessarly
    evaluated with the same environment everytime, this won't be determinist.

    When the init states are different, this is not purely stochastic. This
    choice was made for performance reason, as the reset function of brax envs
    is quite time consuming. If pure stochasticity of the environment is needed
    for a use case, please open an issue.

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
        random_key
    )
