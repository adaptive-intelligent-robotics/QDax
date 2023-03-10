from functools import partial
from typing import Any, Callable, Tuple

import brax.envs
import flax.linen as nn
import jax
import jax.numpy as jnp
from brax.envs import State as EnvState
from flax.struct import PyTreeNode

from qdax.core.neuroevolution.buffers.buffer import Transition
from qdax.types import Genotype, Params, RNGKey


class TrainingState(PyTreeNode):
    """The state of a training process. Can be used to store anything
    that is useful for a training process. This object is used in the
    package to store all stateful object necessary for training an agent
    that learns how to act in an MDP.
    """

    pass


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
        A new state, the experienced transition.
    """

    def _scan_play_step_fn(
        carry: Tuple[EnvState, Params, RNGKey], unused_arg: Any
    ) -> Tuple[Tuple[EnvState, Params, RNGKey], Transition]:
        env_state, policy_params, random_key, transitions = play_step_fn(*carry)
        return (env_state, policy_params, random_key), transitions

    (state, _, _), transitions = jax.lax.scan(
        _scan_play_step_fn,
        (init_state, policy_params, random_key),
        (),
        length=episode_length,
    )
    return state, transitions


@jax.jit
def get_first_episode(transition: Transition) -> Transition:
    """Extracts the first episode from a batch of transitions, returns the batch of
    transitions that is masked with nans except for the first episode.
    """

    dones = jnp.roll(transition.dones, 1, axis=0).at[0].set(0)
    mask = 1 - jnp.clip(jnp.cumsum(dones, axis=0), 0, 1)

    def mask_episodes(x: jnp.ndarray) -> jnp.ndarray:
        # the double transpose trick is here to allow easy broadcasting
        return jnp.where(mask.T, x.T, jnp.nan * jnp.ones_like(x).T).T

    return jax.tree_map(mask_episodes, transition)  # type: ignore


def init_population_controllers(
    policy_network: nn.Module,
    env: brax.envs.Env,
    batch_size: int,
    random_key: RNGKey,
) -> Tuple[Genotype, RNGKey]:
    """
    Initializes the population of controllers using a policy_network.

    Args:
        policy_network: The policy network structure used for creating policy
            controllers.
        env: the BRAX environment.
        batch_size: the number of environments we play simultaneously.
        random_key: a JAX random key.

    Returns:
        A tuple of the initial population and the new random key.
    """
    random_key, subkey = jax.random.split(random_key)

    keys = jax.random.split(subkey, num=batch_size)
    fake_batch = jnp.zeros(shape=(batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    return init_variables, random_key
