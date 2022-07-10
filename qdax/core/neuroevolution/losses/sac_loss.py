from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from brax.training.distribution import ParametricDistribution

from qdax.core.neuroevolution.buffers.buffer import Transition
from qdax.types import Action, Observation, Params, RNGKey


def make_sac_loss_fn(
    policy_fn: Callable[[Params, Observation], jnp.ndarray],
    critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
    parametric_action_distribution: ParametricDistribution,
    reward_scaling: float,
    discount: float,
    action_size: int,
) -> Tuple[
    Callable[[jnp.ndarray, Params, Transition, RNGKey], jnp.ndarray],
    Callable[[Params, Params, jnp.ndarray, Transition, RNGKey], jnp.ndarray],
    Callable[[Params, Params, Params, Transition, RNGKey], jnp.ndarray],
]:
    """Creates the loss used in SAC.

    Args:
        policy_fn: the apply function of the policy
        critic_fn: the apply function of the critic
        parametric_action_distribution: the distribution over actions
        reward_scaling: a multiplicative factor to the reward
        discount: the discount factor
        action_size: the size of the environment's action space

    Returns:
        the loss of the entropy parameter auto-tuning
        the loss of the policy
        the loss of the critic
    """

    @jax.jit
    def _policy_loss_fn(
        policy_params: Params,
        critic_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        random_key: RNGKey,
    ) -> jnp.ndarray:

        dist_params = policy_fn(policy_params, transitions.obs)
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, random_key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)
        q_action = critic_fn(critic_params, transitions.obs, action)
        min_q = jnp.min(q_action, axis=-1)
        actor_loss = alpha * log_prob - min_q

        return jnp.mean(actor_loss)

    @jax.jit
    def _critic_loss_fn(
        critic_params: Params,
        policy_params: Params,
        target_critic_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        random_key: RNGKey,
    ) -> jnp.ndarray:

        q_old_action = critic_fn(critic_params, transitions.obs, transitions.actions)
        next_dist_params = policy_fn(policy_params, transitions.next_obs)
        next_action = parametric_action_distribution.sample_no_postprocessing(
            next_dist_params, random_key
        )
        next_log_prob = parametric_action_distribution.log_prob(
            next_dist_params, next_action
        )
        next_action = parametric_action_distribution.postprocess(next_action)
        next_q = critic_fn(target_critic_params, transitions.next_obs, next_action)

        next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob

        target_q = jax.lax.stop_gradient(
            transitions.rewards * reward_scaling
            + (1.0 - transitions.dones) * discount * next_v
        )

        q_error = q_old_action - jnp.expand_dims(target_q, -1)

        q_error *= jnp.expand_dims(1 - transitions.truncations, -1)

        q_loss = 0.5 * jnp.mean(jnp.square(q_error))

        return q_loss

    target_entropy = -0.5 * action_size

    @jax.jit
    def _alpha_loss_fn(
        log_alpha: jnp.ndarray,
        policy_params: Params,
        transitions: Transition,
        random_key: RNGKey,
    ) -> jnp.ndarray:
        """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""

        dist_params = policy_fn(policy_params, transitions.obs)
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, random_key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)
        alpha = jnp.exp(log_alpha)
        alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)

        loss = jnp.mean(alpha_loss)
        return loss

    return _alpha_loss_fn, _policy_loss_fn, _critic_loss_fn
