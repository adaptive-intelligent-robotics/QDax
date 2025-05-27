"""Implements the PG Emitter from PGA-ME algorithm in jax for brax environments,
based on:
https://hal.archives-ouvertes.fr/hal-03135723v2/file/PGA_MAP_Elites_GECCO.pdf
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp

from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.emitters.repertoire_selectors.selector import Selector
from qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from qdax.core.neuroevolution.losses.td3_loss import make_td3_loss_fn
from qdax.core.neuroevolution.networks.networks import QModule
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey
from qdax.tasks.brax.v1.envs.base_env import QDEnv


@dataclass
class QualityPGConfig:
    """Configuration for QualityPG Emitter"""

    env_batch_size: int = 100
    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100

    # TD3 params
    replay_buffer_size: int = 1000000
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-3
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 256
    soft_tau_update: float = 0.005
    policy_delay: int = 2


class QualityPGEmitterState(EmitterState):
    """Contains training state for the learner."""

    critic_params: Params
    critic_optimizer_state: optax.OptState
    actor_params: Params
    actor_opt_state: optax.OptState
    target_critic_params: Params
    target_actor_params: Params
    replay_buffer: ReplayBuffer
    key: RNGKey


class QualityPGEmitter(Emitter):
    """
    A policy gradient emitter used to implement the Policy Gradient Assisted MAP-Elites
    (PGA-Map-Elites) algorithm.
    """

    def __init__(
        self,
        config: QualityPGConfig,
        policy_network: nn.Module,
        env: QDEnv,
        selector: Optional[Selector] = None,
    ) -> None:
        self._config = config
        self._env = env
        self._selector = selector
        self._actor_critic_iterations = int(
            config.num_critic_training_steps / config.policy_delay
        )  # actor and critic training are packed into a single function

        # Init Critics
        critic_network = QModule(
            n_critics=2, hidden_layer_sizes=self._config.critic_hidden_layer_size
        )
        self._critic_network = critic_network

        # Set up the losses and optimizers - return the opt states
        self._policy_loss_fn, self._critic_loss_fn = make_td3_loss_fn(
            policy_fn=policy_network.apply,
            critic_fn=critic_network.apply,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            noise_clip=self._config.noise_clip,
            policy_noise=self._config.policy_noise,
        )

        # Init optimizers
        self._actor_optimizer = optax.adam(
            learning_rate=self._config.actor_learning_rate
        )
        self._critic_optimizer = optax.adam(
            learning_rate=self._config.critic_learning_rate
        )
        self._policies_optimizer = optax.adam(
            learning_rate=self._config.policy_learning_rate
        )

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._config.env_batch_size

    @property
    def use_all_data(self) -> bool:
        """Whether to use all data or not when used along other emitters.

        QualityPGEmitter uses the transitions from the genotypes that were generated
        by other emitters.
        """
        return True

    def init(
        self,
        key: RNGKey,
        repertoire: Repertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> QualityPGEmitterState:
        """Initializes the emitter state.

        Args:
            genotypes: The initial population.
            key: A random key.

        Returns:
            The initial state of the PGAMEEmitter.
        """

        observation_size = self._env.observation_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_length

        # Initialise critic, greedy actor and population
        key, subkey = jax.random.split(key)
        fake_obs = jnp.zeros(shape=(observation_size,))
        fake_action = jnp.zeros(shape=(action_size,))
        critic_params = self._critic_network.init(
            subkey, obs=fake_obs, actions=fake_action
        )
        target_critic_params = jax.tree.map(lambda x: x, critic_params)

        actor_params = jax.tree.map(lambda x: x[0], genotypes)
        target_actor_params = jax.tree.map(lambda x: x[0], genotypes)

        # Prepare init optimizer states
        critic_optimizer_state = self._critic_optimizer.init(critic_params)
        actor_optimizer_state = self._actor_optimizer.init(actor_params)

        # Initialize replay buffer
        dummy_transition = QDTransition.init_dummy(
            observation_dim=observation_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )

        replay_buffer = ReplayBuffer.init(
            buffer_size=self._config.replay_buffer_size, transition=dummy_transition
        )

        # get the transitions out of the dictionary
        assert "transitions" in extra_scores.keys(), "Missing transitions or wrong key"
        transitions = extra_scores["transitions"]

        # add transitions in the replay buffer
        replay_buffer = replay_buffer.insert(transitions)

        # Initial training state
        emitter_state = QualityPGEmitterState(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            actor_params=actor_params,
            actor_opt_state=actor_optimizer_state,
            target_critic_params=target_critic_params,
            target_actor_params=target_actor_params,
            replay_buffer=replay_buffer,
            key=key,
        )

        return emitter_state

    def emit(  # type: ignore
        self,
        repertoire: GARepertoire,
        emitter_state: QualityPGEmitterState,
        key: RNGKey,
    ) -> Tuple[Genotype, ExtraScores]:
        """Do a step of PG emission.

        Args:
            repertoire: the current repertoire of genotypes
            emitter_state: the state of the emitter used
            key: a random key

        Returns:
            A batch of offspring, a empty dict for signature.
        """

        batch_size = self._config.env_batch_size

        # sample parents
        mutation_pg_batch_size = int(batch_size - 1)
        parents = repertoire.select(
            key, mutation_pg_batch_size, selector=self._selector
        ).genotypes

        # apply the pg mutation
        offsprings_pg = self.emit_pg(emitter_state, parents)

        # get the actor (greedy actor)
        offspring_actor = self.emit_actor(emitter_state)

        # add dimension for concatenation
        offspring_actor = jax.tree.map(
            lambda x: jnp.expand_dims(x, axis=0), offspring_actor
        )
        # gather offspring
        genotypes = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            offsprings_pg,
            offspring_actor,
        )

        return genotypes, {}

    def emit_pg(
        self, emitter_state: QualityPGEmitterState, parents: Genotype
    ) -> Genotype:
        """Emit the offsprings generated through pg mutation.

        Args:
            emitter_state: current emitter state, contains critic and
                replay buffer.
            parents: the parents selected to be applied gradients in order
                to mutate towards better performance.

        Returns:
            A new set of offsprings.
        """

        # create a batch of policy optimizer states
        policy_opt_states = jax.vmap(self._policies_optimizer.init)(parents)

        # prepare the batched policy update function with vmapping
        batched_policy_update_fn = jax.vmap(
            partial(self._update_policy, critic_params=emitter_state.critic_params),
            in_axes=(0, 0, None),
        )

        def scan_update_policies(
            carry: Tuple[Params, optax.OptState, RNGKey],
            _: None,
        ) -> Tuple[Tuple[Params, optax.OptState, RNGKey], Any]:

            # Unpack the carry
            (policy_params, policy_opt_state, key) = carry
            key, subkey = jax.random.split(key)

            # sample a mini-batch of data from the replay-buffer
            transitions = emitter_state.replay_buffer.sample(
                subkey, self._config.batch_size
            )
            (
                new_policy_params,
                new_policy_opt_states,
            ) = batched_policy_update_fn(policy_params, policy_opt_state, transitions)
            return (new_policy_params, new_policy_opt_states, key), ()

        (
            final_policy_params,
            final_policy_opt_state,
            final_key,
        ), _ = jax.lax.scan(
            scan_update_policies,
            (parents, policy_opt_states, emitter_state.key),
            length=self._config.num_pg_training_steps,
        )

        return final_policy_params

    def emit_actor(self, emitter_state: QualityPGEmitterState) -> Genotype:
        """Emit the greedy actor.

        Simply needs to be retrieved from the emitter state.

        Args:
            emitter_state: the current emitter state, it stores the
                greedy actor.

        Returns:
            The parameters of the actor.
        """
        return emitter_state.actor_params

    def state_update(  # type: ignore
        self,
        emitter_state: QualityPGEmitterState,
        repertoire: Optional[Repertoire],
        genotypes: Optional[Genotype],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> QualityPGEmitterState:
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
        emitter_state = emitter_state.replace(
            replay_buffer=emitter_state.replay_buffer.insert(transitions)
        )
        # Conduct Actor-Critic training
        final_emitter_state, _ = jax.lax.scan(
            self._scan_actor_critic_training,
            emitter_state,
            length=self._actor_critic_iterations,
        )

        return final_emitter_state  # type: ignore

    def _scan_update_critic(
        self,
        carry: Tuple[Params, Params, optax.OptState, Params, RNGKey],
        transitions: QDTransition,
    ) -> Tuple[Tuple[Params, Params, optax.OptState, Params, RNGKey], Any]:
        """A scan-ready function to update the critic network parameters
        with one gradient step.

        Args:
            carry: packed carry containing critic parameters, target critic
                parameters, critic optimiser state, target actor parameters,
                and the random key.
            transitions: a mini-batch of QDTransitions for gradient computation

        Returns:
            new_carry: new carry containing updated parameters (target actor
                parameters is unchanged).
            empty tuple: compulsory signature for jax.lax.scan
        """
        # unpack the carry
        (
            critic_params,
            target_critic_params,
            critic_opt_state,
            target_actor_params,
            key,
        ) = carry

        # compute the critic gradients
        key, subkey = jax.random.split(key)
        critic_gradient = jax.grad(self._critic_loss_fn)(
            critic_params,
            target_actor_params,
            target_critic_params,
            transitions,
            subkey,
        )
        # update critic
        critic_updates, new_critic_opt_state = self._critic_optimizer.update(
            critic_gradient, critic_opt_state
        )
        new_critic_params = optax.apply_updates(critic_params, critic_updates)

        # Soft update of target critic network
        new_target_critic_params = jax.tree.map(
            lambda x, y: (1.0 - self._config.soft_tau_update) * x
            + self._config.soft_tau_update * y,
            target_critic_params,
            new_critic_params,
        )
        # pack into new carry
        new_carry = (
            new_critic_params,
            new_target_critic_params,
            new_critic_opt_state,
            target_actor_params,
            key,
        )
        return new_carry, ()

    def _update_actor(
        self,
        actor_params: Params,
        actor_opt_state: optax.OptState,
        target_actor_params: Params,
        critic_params: Params,
        transitions: QDTransition,
    ) -> Tuple[optax.OptState, Params, Params]:
        """Function to update the actor with one Policy-Gradient step.

        Args:
            actor_params: neural network parameters of the actor.
            actor_opt_state: optimiser state of the actor.
            target_actor_params: target actor parameters (used in TD3 to
                smooth the actor-critic learning).
            critic_params: the parameters of the critic networks which plays
                as a differentiable target.
            transitions: a mini-batch of QDTransitions.

        Returns:
            new_actor_params: new actor parameters after taking the PG step
            new_target_actor_params: new target actor parameters (soft-tau update)
            new_actor_opt_state: updated optimiser state
        """

        # Update greedy actor
        policy_gradient = jax.grad(self._policy_loss_fn)(
            actor_params,
            critic_params,
            transitions,
        )
        (
            policy_updates,
            new_actor_opt_state,
        ) = self._actor_optimizer.update(policy_gradient, actor_opt_state)
        new_actor_params = optax.apply_updates(actor_params, policy_updates)

        # Soft update of target greedy actor
        new_target_actor_params = jax.tree.map(
            lambda x, y: (1.0 - self._config.soft_tau_update) * x
            + self._config.soft_tau_update * y,
            target_actor_params,
            new_actor_params,
        )

        return new_actor_params, new_target_actor_params, new_actor_opt_state

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

    def _update_policy(
        self,
        policy_params: Params,
        policy_opt_state: optax.OptState,
        transitions: QDTransition,
        critic_params: Params,
    ) -> Tuple[optax.OptState, Params]:
        """
        Perform one step of PG update on the off-spring policy.
        This function is vmapped to mutate the entire batch of off-springs
        in parallel.

        Args:
            policy_params: the parameters of the policy.
            policy_opt_state: the optimiser state of the policy.
            transitions: a mini-batch of transitions for gradient computation
            critic_params: the parameters of the critic networks serving as
                a differentiable target. This is fixed in each iteration.

        Returns:
            new_policy_params: new policy parameters
            new_policy_opt_state: updated optimiser state
        """

        # Compute the policy gradient
        policy_gradient = jax.grad(self._policy_loss_fn)(
            policy_params,
            critic_params,
            transitions,
        )
        # Apply the update on the policy
        (
            policy_updates,
            new_policy_opt_state,
        ) = self._policies_optimizer.update(policy_gradient, policy_opt_state)
        new_policy_params = optax.apply_updates(policy_params, policy_updates)

        return new_policy_params, new_policy_opt_state
