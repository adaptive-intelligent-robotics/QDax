"""Variants of DIAYN that enable to use the extrinsic reward of
the environment to discover supervised skills. Variants are:
- SMERL DIAYN (sums rewards depending on a condition)
- DIAYN SUM (simply sums the rewards)
"""

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import jax

from qdax.baselines.diayn import DIAYN, DiaynConfig, DiaynTrainingState
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.buffers.trajectory_buffer import TrajectoryBuffer
from qdax.types import Metrics, Reward


@dataclass
class DiaynSmerlConfig(DiaynConfig):
    """Configuration for the SMERL_DIAYN algorithm"""

    diversity_reward_scale: float = 1.0
    smerl_target: Optional[float] = 1e3
    smerl_margin: Optional[float] = 1e2


class DIAYNSMERL(DIAYN):
    """DIAYNSMERL refers to a family of methods that combine the DIAYN's diversity
    reward with some environment `extrinsic` reward, using SMERL method, see
    https://arxiv.org/abs/2010.14484.

    Most methods are inherited from the DIAYN algorithm, the only change is the
    way the reward is computed (a combination of the DIAYN reward and
    the `extrinsic` reward).
    """

    def __init__(self, config: DiaynSmerlConfig, action_size: int):
        super(DIAYNSMERL, self).__init__(config, action_size)
        self._config: DiaynSmerlConfig = config

    @partial(jax.jit, static_argnames=("self",))
    def _compute_reward(
        self,
        transition: QDTransition,
        training_state: DiaynTrainingState,
        returns: Reward,
    ) -> Reward:
        """Computes the reward to train the networks.

        Args:
            transition: a batch of transitions from the replay buffer
            training_state: the current training state
            returns: an array containing the episode's return for every sample

        Returns:
            the combined reward
        """

        # Compute diversity reward
        diversity_rewards = self._compute_diversity_reward(
            transition=transition,
            discriminator_params=training_state.discriminator_params,
            add_log_p_z=True,
        )

        # Compute SMERL reward
        assert (
            self._config.smerl_target is not None
            and self._config.smerl_margin is not None
        ), "Missing SMERL target and margin values"

        # is the return good enough to consider the diversity reward
        accept = returns >= self._config.smerl_target - self._config.smerl_margin

        # compute the new reward (r_extrinsic + accept * diversity_scale * r_diversity)
        rewards = (
            transition.rewards
            + accept * self._config.diversity_reward_scale * diversity_rewards
        )

        return rewards

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        training_state: DiaynTrainingState,
        replay_buffer: TrajectoryBuffer,
    ) -> Tuple[DiaynTrainingState, TrajectoryBuffer, Metrics]:
        """Performs a training step to update the policy, the critic and the
        discriminator parameters.

        Args:
            training_state: the current DIAYN training state
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
            state_desc = samples.obs[:, : -self._config.num_skills]
            next_state_desc = samples.next_obs[:, : -self._config.num_skills]
            samples = samples.replace(
                state_desc=state_desc, next_state_desc=next_state_desc
            )

        # Compute the rewards
        rewards = self._compute_reward(samples, training_state, returns)

        samples = samples.replace(rewards=rewards)

        new_training_state, metrics = self._update_networks(
            training_state, transitions=samples
        )

        return new_training_state, replay_buffer, metrics
