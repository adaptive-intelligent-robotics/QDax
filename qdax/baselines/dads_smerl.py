"""Variants of DADS that enable to use the extrinsic reward of
the environment to discover supervised skills. Variants are:
- SMERL DADS (sums rewards depending on a condition)
- DADS SUM (simply sums the rewards)
"""

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import jax

from qdax.baselines.dads import DADS, DadsConfig, DadsTrainingState
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.buffers.trajectory_buffer import TrajectoryBuffer
from qdax.core.neuroevolution.normalization_utils import normalize_with_rmstd
from qdax.types import Metrics, Reward


@dataclass
class DadsSmerlConfig(DadsConfig):
    """Configuration for the SMERL_DADS algorithm"""

    diversity_reward_scale: float = 1.0
    smerl_target: Optional[float] = 1e3
    smerl_margin: Optional[float] = 1e2


class DADSSMERL(DADS):
    """DADSSMERL refers to a family of methods that combine the DADS's diversity
    reward with some environment `extrinsic` reward, using the proper SMERL method,
    see https://arxiv.org/abs/2010.14484.

    Most of the methods are inherited from the DADS algorithm, the only change is
    the way the reward is computed (a combination of the DADS reward and the `extrinsic`
    reward).
    """

    def __init__(self, config: DadsSmerlConfig, action_size: int, descriptor_size: int):
        super(DADSSMERL, self).__init__(config, action_size, descriptor_size)
        self._config: DadsSmerlConfig = config

    @partial(jax.jit, static_argnames=("self",))
    def _compute_reward(
        self,
        transition: QDTransition,
        training_state: DadsTrainingState,
        returns: Reward,
    ) -> Reward:
        """Computes the reward to train the networks.

        Args:
            transition: a batch of transitions from the replay buffer
            training_state: the current training state

        Returns:
            the reward
        """

        diversity_rewards = self._compute_diversity_reward(
            transition=transition, training_state=training_state
        )
        # Compute SMERL reward (r_extrinsic + accept * diversity_scale * r_diversity)
        assert (
            self._config.smerl_target is not None
            and self._config.smerl_margin is not None
        ), "Missing SMERL target and margin values"

        accept = returns >= self._config.smerl_target - self._config.smerl_margin
        rewards = (
            transition.rewards
            + accept * self._config.diversity_reward_scale * diversity_rewards
        )

        return rewards

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        training_state: DadsTrainingState,
        replay_buffer: TrajectoryBuffer,
    ) -> Tuple[DadsTrainingState, TrajectoryBuffer, Metrics]:
        """Performs a training step to update the policy, the critic and the
        dynamics network parameters.

        Args:
            training_state: the current DADS training state
            replay_buffer: the replay buffer

        Returns:
            the updated DIAYN training state
            the replay buffer
            the training metrics
        """

        # Sample a batch of transitions in the buffer
        random_key = training_state.random_key
        samples, returns, random_key = replay_buffer.sample_with_returns(
            random_key,
            sample_size=self._config.batch_size,
        )

        # Optionally replace the state descriptor by the observation
        if self._config.descriptor_full_state:
            _state_desc = samples.obs[:, : -self._config.num_skills]
            _next_state_desc = samples.next_obs[:, : -self._config.num_skills]
            samples = samples.replace(
                state_desc=_state_desc, next_state_desc=_next_state_desc
            )

        # Compute the reward
        rewards = self._compute_reward(
            transition=samples, training_state=training_state, returns=returns
        )

        # Compute the target and optionally normalize it for the training
        if self._config.normalize_target:
            next_state_desc = normalize_with_rmstd(
                samples.next_state_desc - samples.state_desc,
                training_state.normalization_running_stats,
            )

        else:
            next_state_desc = samples.next_state_desc - samples.state_desc

        # Update the transitions
        samples = samples.replace(next_state_desc=next_state_desc, rewards=rewards)

        new_training_state, metrics = self._update_networks(
            training_state, transitions=samples
        )
        return new_training_state, replay_buffer, metrics
