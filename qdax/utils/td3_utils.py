from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from qdax.types import Params, RNGKey
from qdax.utils.mdp_utils import Transition
from qdax.utils.networks import MLP, QModule


def make_td3_loss_fn(
    policy_network: MLP,
    q_network: QModule,
    reward_scaling: float,
    discount: float,
    noise_clip: float,
    policy_noise: float,
) -> Tuple[Callable, Callable]:
    """Creates the loss functions for TD3"""

    @jax.jit
    def _policy_loss_fn(
        policy_params: Params,
        q_params: Params,
        transitions: Transition,
    ) -> jnp.ndarray:
        """Policy loss function for TD3 agent"""

        action = policy_network.apply(policy_params, transitions.obs)
        q_value = q_network.apply(q_params, obs=transitions.obs, actions=action)
        q1_action = q_value[:, 0]
        policy_loss = -jnp.mean(q1_action)
        return policy_loss

    @jax.jit
    def _critic_loss_fn(
        q_params: Params,
        target_policy_params: Params,
        target_q_params: Params,
        transitions: Transition,
        key: RNGKey,
    ) -> jnp.ndarray:
        """Critics loss function for TD3 agent"""
        noise = (
            jax.random.normal(key, shape=transitions.actions.shape) * policy_noise
        ).clip(-noise_clip, noise_clip)

        next_action = (
            policy_network.apply(target_policy_params, transitions.next_obs) + noise
        ).clip(-1.0, 1.0)
        next_q = q_network.apply(
            target_q_params, obs=transitions.next_obs, actions=next_action
        )
        next_v = jnp.min(next_q, axis=-1)
        target_q = jax.lax.stop_gradient(
            transitions.rewards * reward_scaling
            + (1.0 - transitions.dones) * discount * next_v
        )
        q_old_action = q_network.apply(
            q_params,
            obs=transitions.obs,
            actions=transitions.actions,
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)

        # Better bootstrapping for truncated episodes.
        q_error *= jnp.expand_dims(1 - transitions.truncations, -1)

        q_loss = 0.5 * jnp.mean(jnp.square(q_error))
        return q_loss

    return _policy_loss_fn, _critic_loss_fn
