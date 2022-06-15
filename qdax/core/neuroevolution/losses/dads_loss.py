from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from brax.training.distribution import ParametricDistribution

from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.losses.sac_loss import make_sac_loss_fn
from qdax.types import Action, Observation, Params, RNGKey, Skill, StateDescriptor


def make_dads_loss_fn(
    policy_fn: Callable[[Params, Observation], jnp.ndarray],
    critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
    dynamics_fn: Callable[
        [Params, StateDescriptor, Skill, StateDescriptor], jnp.ndarray
    ],
    parametric_action_distribution: ParametricDistribution,
    reward_scaling: float,
    discount: float,
    action_size: int,
    num_skills: int,
) -> Tuple[
    Callable[[jnp.ndarray, Params, QDTransition, RNGKey], jnp.ndarray],
    Callable[[Params, Params, jnp.ndarray, QDTransition, RNGKey], jnp.ndarray],
    Callable[[Params, Params, Params, QDTransition, RNGKey], jnp.ndarray],
    Callable[[Params, QDTransition, RNGKey], jnp.ndarray],
]:
    """Creates the loss used in DADS.

    Args:
        policy_fn: the apply function of the policy
        critic_fn: the apply function of the critic
        dynamics_fn: the apply function of the dynamics network
        parametric_action_distribution: the distribution over action
        reward_scaling: a multiplicative factor to the reward
        discount: the discount factor
        action_size: the size of the environment's action space
        num_skills: the number of skills set

    Returns:
        the loss of the entropy parameter auto-tuning
        the loss of the policy
        the loss of the critic
        the loss of the dynamics network
    """

    _alpha_loss_fn, _policy_loss_fn, _critic_loss_fn = make_sac_loss_fn(
        policy_fn=policy_fn,
        critic_fn=critic_fn,
        reward_scaling=reward_scaling,
        discount=discount,
        action_size=action_size,
        parametric_action_distribution=parametric_action_distribution,
    )

    @jax.jit
    def _dynamics_loss_fn(
        dynamics_params: Params,
        transitions: QDTransition,
    ) -> jnp.ndarray:
        """Computes the loss used to train the dynamics network.

        Args:
            dynamics_params: the parameters of the neural network
                used to predict the dynamics.
            transitions: the batch of transitions used to train. They
                have been sampled from a replay buffer beforehand.

        Returns:
            The loss obtained on the batch of transitions.
        """

        active_skills = transitions.obs[:, -num_skills:]
        target = transitions.next_state_desc
        log_prob = dynamics_fn(  # type: ignore
            dynamics_params,
            obs=transitions.state_desc,
            skill=active_skills,
            target=target,
        )

        # prevent training on malformed target
        loss = -jnp.mean(log_prob * (1 - transitions.dones))
        return loss

    return _alpha_loss_fn, _policy_loss_fn, _critic_loss_fn, _dynamics_loss_fn
