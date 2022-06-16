from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from brax.training.distribution import ParametricDistribution

from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.losses.sac_loss import make_sac_loss_fn
from qdax.types import Action, Observation, Params, RNGKey, StateDescriptor


def make_diayn_loss_fn(
    policy_fn: Callable[[Params, Observation], jnp.ndarray],
    critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
    discriminator_fn: Callable[[Params, StateDescriptor], jnp.ndarray],
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
    """Creates the loss used in DIAYN.

    Args:
        policy_fn: the apply function of the policy
        critic_fn: the apply function of the critic
        discriminator_fn: the apply function of the discriminator
        parametric_action_distribution: the distribution over actions
        reward_scaling: a multiplicative factor to the reward
        discount: the discount factor
        action_size: the size of the environment's action space
        num_skills: the number of skills set

    Returns:
        the loss of the entropy parameter auto-tuning
        the loss of the policy
        the loss of the critic
        the loss of the discriminator
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
    def _discriminator_loss_fn(
        discriminator_params: Params,
        transitions: QDTransition,
    ) -> jnp.ndarray:
        """Computes the loss used to train the discriminator. The
        discriminator is trained to predict the skill that has been
        used to generate transitions. In our case, skills are one
        hot encoded, the discriminator is hence trained like a
        multi-label classifier, using the categorical cross entropy
        loss.

        In this loss, log softmax outputs the log probabilities for
        each skill. By multiplying by the skills (that are one hot
        vectors), we filter to keep only the log probability of the
        true skill.

        Args:
            discriminator_params: the parameters of the neural network
                used to discriminate the skills.
            transitions: the transitions sampled from the replay buffer.

        Returns:
            The loss of the discriminator on those transitions.
        """

        state_desc = transitions.state_desc
        skills = transitions.obs[:, -num_skills:]
        logits = jnp.sum(
            jax.nn.log_softmax(discriminator_fn(discriminator_params, state_desc))
            * skills,
            axis=1,
        )

        loss = -jnp.mean(logits)
        return loss

    return _alpha_loss_fn, _policy_loss_fn, _critic_loss_fn, _discriminator_loss_fn
