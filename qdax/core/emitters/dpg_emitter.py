""" Implements the Diversity PG inspired by QDPG algorithm in jax for brax environments,
based on: https://arxiv.org/abs/2006.08505
"""
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax
import optax

from qdax.core.containers.archive import Archive
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.qpg_emitter import (
    QualityPGConfig,
    QualityPGEmitter,
    QualityPGEmitterState,
)
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.environments.base_wrappers import QDEnv
from qdax.types import (
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Params,
    Reward,
    RNGKey,
    StateDescriptor,
)


@dataclass
class DiversityPGConfig(QualityPGConfig):
    """Configuration for DiversityPG Emitter"""

    # inherits fields from QualityPGConfig

    # Archive params
    archive_acceptance_threshold: float = 0.1
    archive_max_size: int = 10000


class DiversityPGEmitterState(QualityPGEmitterState):
    """Contains training state for the learner."""

    # inherits from QualityPGEmitterState

    archive: Archive


class DiversityPGEmitter(QualityPGEmitter):
    """
    A diversity policy gradient emitter used to implement QDPG algorithm.

    Please not that the inheritence between DiversityPGEmitter and QualityPGEmitter
    could be increased with changes in the way transitions samples are handled in
    the QualityPGEmitter. But this would modify the computation/memory strategy of the
    current implementation. Hence, we won't apply this yet and will discuss this with
    the development team.
    """

    def __init__(
        self,
        config: DiversityPGConfig,
        policy_network: nn.Module,
        env: QDEnv,
        score_novelty: Callable[[Archive, StateDescriptor], Reward],
    ) -> None:

        # usual init operations from PGAME
        super().__init__(config, policy_network, env)

        self._config: DiversityPGConfig = config

        # define scoring function
        self._score_novelty = score_novelty

    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[DiversityPGEmitterState, RNGKey]:
        """Initializes the emitter state.

        Args:
            init_genotypes: The initial population.
            random_key: A random key.

        Returns:
            The initial state of the PGAMEEmitter, a new random key.
        """

        # init elements of diversity emitter state with QualityEmitterState.init()
        diversity_emitter_state, random_key = super().init(init_genotypes, random_key)

        # store elements in a dictionary
        attributes_dict = vars(diversity_emitter_state)

        # init archive
        archive = Archive.create(
            acceptance_threshold=self._config.archive_acceptance_threshold,
            state_descriptor_size=self._env.state_descriptor_length,
            max_size=self._config.archive_max_size,
        )

        # init emitter state
        emitter_state = DiversityPGEmitterState(
            # retrieve all attributes from the QualityPGEmitterState
            **attributes_dict,
            # add the last element: archive
            archive=archive,
        )

        return emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: DiversityPGEmitterState,
        repertoire: Optional[Repertoire],
        genotypes: Optional[Genotype],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> DiversityPGEmitterState:
        """This function gives an opportunity to update the emitter state
        after the genotypes have been scored.

        Here it is used to fill the Replay Buffer with the transitions
        from the scoring of the genotypes, and then the training of the
        critic/actor happens. Hence the params of critic/actor are updated,
        as well as their optimizer states.

        Args:
            emitter_state: current emitter state.
            repertoire: the current genotypes repertoire
            genotypes: unused here - but compulsory in the signature.
            fitnesses: unused here - but compulsory in the signature.
            descriptors: unused here - but compulsory in the signature.
            extra_scores: extra information coming from the scoring function,
                this contains the transitions added to the replay buffer.

        Returns:
            New emitter state where the replay buffer has been filled with
            the new experienced transitions.
        """
        # get the transitions out of the dictionary
        assert "transitions" in extra_scores.keys(), "Missing transitions or wrong key"
        transitions = extra_scores["transitions"]

        # add transitions in the replay buffer
        replay_buffer = emitter_state.replay_buffer.insert(transitions)
        emitter_state = emitter_state.replace(replay_buffer=replay_buffer)

        archive = emitter_state.archive.insert(transitions.state_desc)

        def scan_train_critics(
            carry: DiversityPGEmitterState, transitions: QDTransition
        ) -> Tuple[DiversityPGEmitterState, Any]:
            emitter_state = carry
            new_emitter_state = self._train_critics(emitter_state, transitions)
            return new_emitter_state, ()

        # sample transitions
        (transitions, random_key,) = emitter_state.replay_buffer.sample(
            random_key=emitter_state.random_key,
            sample_size=self._config.num_critic_training_steps
            * self._config.batch_size,
        )

        # update the rewards - diversity rewards
        state_descriptors = transitions.state_desc
        diversity_rewards = self._score_novelty(archive, state_descriptors)
        transitions = transitions.replace(rewards=diversity_rewards)

        # reshape the transitions
        transitions = jax.tree_util.tree_map(
            lambda x: x.reshape(
                (
                    self._config.num_critic_training_steps,
                    self._config.batch_size,
                )
                + x.shape[1:]
            ),
            transitions,
        )

        # Train critics and greedy actor
        emitter_state, _ = jax.lax.scan(
            scan_train_critics,
            emitter_state,
            (transitions),
            length=self._config.num_critic_training_steps,
        )

        emitter_state = emitter_state.replace(archive=archive)

        return emitter_state  # type: ignore

    @partial(jax.jit, static_argnames=("self",))
    def _train_critics(
        self, emitter_state: DiversityPGEmitterState, transitions: QDTransition
    ) -> DiversityPGEmitterState:
        """Apply one gradient step to critics and to the greedy actor
        (contained in carry in training_state), then soft update target critics
        and target greedy actor.

        Those updates are very similar to those made in TD3.

        Args:
            emitter_state: actual emitter state

        Returns:
            New emitter state where the critic and the greedy actor have been
            updated. Optimizer states have also been updated in the process.
        """

        # Update Critic
        (
            critic_optimizer_state,
            critic_params,
            target_critic_params,
            random_key,
        ) = self._update_critic(
            critic_params=emitter_state.critic_params,
            target_critic_params=emitter_state.target_critic_params,
            target_actor_params=emitter_state.target_actor_params,
            critic_optimizer_state=emitter_state.critic_optimizer_state,
            transitions=transitions,
            random_key=emitter_state.random_key,
        )

        # Update greedy policy
        (policy_optimizer_state, actor_params, target_actor_params,) = jax.lax.cond(
            emitter_state.steps % self._config.policy_delay == 0,
            lambda x: self._update_actor(*x),
            lambda _: (
                emitter_state.actor_opt_state,
                emitter_state.actor_params,
                emitter_state.target_actor_params,
            ),
            operand=(
                emitter_state.actor_params,
                emitter_state.actor_opt_state,
                emitter_state.target_actor_params,
                emitter_state.critic_params,
                transitions,
            ),
        )

        # Create new training state
        new_emitter_state = emitter_state.replace(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            actor_params=actor_params,
            actor_opt_state=policy_optimizer_state,
            target_critic_params=target_critic_params,
            target_actor_params=target_actor_params,
            random_key=random_key,
            steps=emitter_state.steps + 1,
            replay_buffer=emitter_state.replay_buffer,
        )

        return new_emitter_state  # type: ignore

    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_pg(
        self,
        policy_params: Genotype,
        emitter_state: DiversityPGEmitterState,
    ) -> Genotype:
        """Apply pg mutation to a policy via multiple steps of gradient descent.

        Args:
            policy_params: a policy, supposed to be a differentiable neural
                network.
            emitter_state: the current state of the emitter, containing among others,
                the replay buffer, the critic.

        Returns:
            the updated params of the neural network.
        """

        # Define new policy optimizer state
        policy_optimizer_state = self._policies_optimizer.init(policy_params)

        def scan_train_policy(
            carry: Tuple[DiversityPGEmitterState, Genotype, optax.OptState],
            transitions: QDTransition,
        ) -> Tuple[Tuple[DiversityPGEmitterState, Genotype, optax.OptState], Any]:
            emitter_state, policy_params, policy_optimizer_state = carry
            (
                new_emitter_state,
                new_policy_params,
                new_policy_optimizer_state,
            ) = self._train_policy(
                emitter_state,
                policy_params,
                policy_optimizer_state,
                transitions,
            )
            return (
                new_emitter_state,
                new_policy_params,
                new_policy_optimizer_state,
            ), ()

        # sample transitions
        transitions, _random_key = emitter_state.replay_buffer.sample(
            random_key=emitter_state.random_key,
            sample_size=self._config.num_pg_training_steps * self._config.batch_size,
        )

        # update the rewards - diversity rewards
        state_descriptors = transitions.state_desc
        diversity_rewards = self._score_novelty(
            emitter_state.archive, state_descriptors
        )
        transitions = transitions.replace(rewards=diversity_rewards)

        # reshape the transitions
        transitions = jax.tree_util.tree_map(
            lambda x: x.reshape(
                (
                    self._config.num_pg_training_steps,
                    self._config.batch_size,
                )
                + x.shape[1:]
            ),
            transitions,
        )

        (emitter_state, policy_params, policy_optimizer_state,), _ = jax.lax.scan(
            scan_train_policy,
            (emitter_state, policy_params, policy_optimizer_state),
            (transitions),
            length=self._config.num_pg_training_steps,
        )

        return policy_params

    @partial(jax.jit, static_argnames=("self",))
    def _train_policy(
        self,
        emitter_state: DiversityPGEmitterState,
        policy_params: Params,
        policy_optimizer_state: optax.OptState,
        transitions: QDTransition,
    ) -> Tuple[DiversityPGEmitterState, Params, optax.OptState]:
        """Apply one gradient step to a policy (called policies_params).

        Args:
            emitter_state: current state of the emitter.
            policy_params: parameters corresponding to the weights and bias of
                the neural network that defines the policy.

        Returns:
            The new emitter state and new params of the NN.
        """

        # update policy
        policy_optimizer_state, policy_params = self._update_policy(
            critic_params=emitter_state.critic_params,
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            transitions=transitions,
        )

        return emitter_state, policy_params, policy_optimizer_state
