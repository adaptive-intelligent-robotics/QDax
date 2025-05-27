""" Implements the Diversity PG inspired by QDPG algorithm in jax for brax environments,
based on: https://arxiv.org/abs/2006.08505
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
from jax import numpy as jnp

from qdax.core.containers.archive import Archive
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.qpg_emitter import (
    QualityPGConfig,
    QualityPGEmitter,
    QualityPGEmitterState,
)
from qdax.core.emitters.repertoire_selectors.selector import Selector
from qdax.custom_types import (
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Reward,
    RNGKey,
    StateDescriptor,
)
from qdax.tasks.brax.v1.envs.base_env import QDEnv


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

    Please note that the inheritance between DiversityPGEmitter and QualityPGEmitter
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
        selector: Optional[Selector] = None,
    ) -> None:

        # usual init operations from PGAME
        super().__init__(config, policy_network, env, selector)

        self._config: DiversityPGConfig = config
        # define scoring function
        self._score_novelty = score_novelty

    def init(
        self,
        key: RNGKey,
        repertoire: Repertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> DiversityPGEmitterState:
        """Initializes the emitter state.

        Args:
            key: A random key.
            repertoire: The initial repertoire.
            genotypes: The initial population.
            fitnesses: The initial fitnesses of the population.
            descriptors: The initial descriptors of the population.
            extra_scores: Extra scores coming from the scoring function.

        Returns:
            The initial state of the PGAMEEmitter.
        """

        # init elements of diversity emitter state with QualityEmitterState.init()
        diversity_emitter_state = super().init(
            key,
            repertoire,
            genotypes,
            fitnesses,
            descriptors,
            extra_scores,
        )

        # store elements in a dictionary
        attributes_dict = vars(diversity_emitter_state)

        # init archive
        archive = Archive.create(
            acceptance_threshold=self._config.archive_acceptance_threshold,
            state_descriptor_size=self._env.state_descriptor_length,
            max_size=self._config.archive_max_size,
        )

        # get the transitions out of the dictionary
        assert "transitions" in extra_scores.keys(), "Missing transitions or wrong key"
        transitions = extra_scores["transitions"]
        archive = archive.insert(transitions.state_desc)

        # init emitter state
        emitter_state = DiversityPGEmitterState(
            # retrieve all attributes from the QualityPGEmitterState
            **attributes_dict,
            # add the last element: archive
            archive=archive,
        )

        return emitter_state

    def state_update(  # type: ignore
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

        # add transitions to the replay buffer and archive
        emitter_state = emitter_state.replace(
            replay_buffer=emitter_state.replay_buffer.insert(transitions),
            archive=emitter_state.archive.insert(transitions.state_desc),
        )

        # Conduct Actor-Critic training
        final_emitter_state, _ = jax.lax.scan(
            self._scan_actor_critic_training,
            emitter_state,
            length=self._actor_critic_iterations,
        )

        return final_emitter_state  # type: ignore

    def _scan_actor_critic_training(
        self, carry: QualityPGEmitterState, _: None
    ) -> Tuple[QualityPGEmitterState, Tuple]:
        """
        Perform a few (policy delay) steps of critic followed by one step of
        actor training, all packed into a single scan-ready function.
        Transition data are sampled step-by-step to promote memory efficiency.

        Args:
            carry: emitter state
            _: None

        Returns:
            new_emitter_state: new emitter state containing updated network
                parameters as the new carry.
            empty tuple: compulsory signature for jax.lax.scan
        """

        emitter_state = carry
        key, subkey = jax.random.split(emitter_state.key)

        # sample transitions
        transitions = emitter_state.replay_buffer.sample(
            subkey,
            self._config.batch_size * (self._config.policy_delay + 1),
        )

        # update the rewards - diversity rewards
        state_descriptors = transitions.state_desc
        diversity_rewards = self._score_novelty(
            emitter_state.archive, state_descriptors
        )
        transitions = transitions.replace(rewards=diversity_rewards)

        transitions = jax.tree.map(
            lambda x: jnp.reshape(
                x,
                (
                    self._config.policy_delay + 1,
                    self._config.batch_size,
                    *x.shape[1:],
                ),
            ),
            transitions,
        )
        # split the transitions for critic and actor
        critic_data = jax.tree.map(lambda x: x[:-1], transitions)
        actor_data = jax.tree.map(lambda x: x[-1], transitions)

        # scan training critics
        (
            new_critic_params,
            new_target_critic_params,
            new_critic_opt_state,
            target_actor_params,
            key,
        ), () = jax.lax.scan(
            self._scan_update_critic,
            (
                emitter_state.critic_params,
                emitter_state.target_critic_params,
                emitter_state.critic_optimizer_state,
                emitter_state.target_actor_params,
                key,
            ),
            critic_data,
        )
        # update the actor with one gradient step
        (new_actor_params, new_target_actor_params, new_actor_opt_state) = (
            self._update_actor(
                emitter_state.actor_params,
                emitter_state.actor_opt_state,
                target_actor_params,
                new_critic_params,
                actor_data,
            )
        )
        new_emitter_state = emitter_state.replace(
            critic_params=new_critic_params,
            critic_optimizer_state=new_critic_opt_state,
            actor_params=new_actor_params,
            actor_opt_state=new_actor_opt_state,
            target_critic_params=new_target_critic_params,
            target_actor_params=new_target_actor_params,
            key=key,
        )

        return new_emitter_state, ()
