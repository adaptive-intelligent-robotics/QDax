"""Implements the DCRL Emitter from DCRL-MAP-Elites algorithm
in JAX for Brax environments.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Tuple

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.neuroevolution.buffers.buffer import DCRLTransition, ReplayBuffer
from qdax.core.neuroevolution.losses.td3_loss import make_td3_loss_dc_fn
from qdax.core.neuroevolution.networks.networks import QModuleDC
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey
from qdax.environments.base_wrappers import QDEnv


@dataclass
class DCRLConfig:
    """Configuration for DCRL Emitter"""

    dcrl_batch_size: int = 64
    ai_batch_size: int = 64
    lengthscale: float = 0.1

    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    num_critic_training_steps: int = 3000
    num_pg_training_steps: int = 150
    batch_size: int = 100
    replay_buffer_size: int = 1_000_000
    discount: float = 0.99
    reward_scaling: float = 1.0
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-3
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    soft_tau_update: float = 0.005
    policy_delay: int = 2


class DCRLEmitterState(EmitterState):
    """Contains training state for the learner."""

    critic_params: Params
    critic_opt_state: optax.OptState
    actor_params: Params
    actor_opt_state: optax.OptState
    target_critic_params: Params
    target_actor_params: Params
    replay_buffer: ReplayBuffer
    key: RNGKey
    steps: jnp.ndarray


class DCRLEmitter(Emitter):
    """
    A descriptor-conditioned reinforcement learning emitter used to implement
    DCRL-MAP-Elites algorithm.
    """

    def __init__(
        self,
        config: DCRLConfig,
        policy_network: nn.Module,
        actor_network: nn.Module,
        env: QDEnv,
    ) -> None:
        self._config = config
        self._env = env
        self._policy_network = policy_network
        self._actor_network = actor_network

        # Init Critics
        critic_network = QModuleDC(
            n_critics=2, hidden_layer_sizes=self._config.critic_hidden_layer_size
        )
        self._critic_network = critic_network

        # Set up the losses and optimizers - return the opt states
        (
            self._policy_loss_fn,
            self._actor_loss_fn,
            self._critic_loss_fn,
        ) = make_td3_loss_dc_fn(
            policy_fn=policy_network.apply,
            actor_fn=actor_network.apply,
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
        return self._config.dcrl_batch_size + self._config.ai_batch_size

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
    ) -> Tuple[DCRLEmitterState, RNGKey]:
        """Initializes the emitter state.

        Args:
            genotypes: The initial population.
            key: A random key.

        Returns:
            The initial state of the PGAMEEmitter, a new random key.
        """

        observation_size = jax.tree_util.tree_leaves(genotypes)[1].shape[1]
        descriptor_size = self._env.behavior_descriptor_length
        action_size = self._env.action_size

        # Initialise critic, greedy actor and population
        key, subkey = jax.random.split(key)
        fake_obs = jnp.zeros(shape=(observation_size,))
        fake_desc = jnp.zeros(shape=(descriptor_size,))
        fake_action = jnp.zeros(shape=(action_size,))

        critic_params = self._critic_network.init(
            subkey, obs=fake_obs, actions=fake_action, desc=fake_desc
        )
        target_critic_params = jax.tree_util.tree_map(lambda x: x, critic_params)

        key, subkey = jax.random.split(key)
        actor_params = self._actor_network.init(subkey, obs=fake_obs, desc=fake_desc)
        target_actor_params = jax.tree_util.tree_map(lambda x: x, actor_params)

        # Prepare init optimizer states
        critic_opt_state = self._critic_optimizer.init(critic_params)
        actor_opt_state = self._actor_optimizer.init(actor_params)

        # Initialize replay buffer
        dummy_transition = DCRLTransition.init_dummy(
            observation_dim=self._env.observation_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )

        replay_buffer = ReplayBuffer.init(
            buffer_size=self._config.replay_buffer_size, transition=dummy_transition
        )

        assert "transitions" in extra_scores.keys(), "Missing transitions or wrong key"
        transitions = extra_scores["transitions"]
        episode_length = transitions.obs.shape[1]

        desc = jnp.repeat(descriptors[:, jnp.newaxis, :], episode_length, axis=1)
        desc_normalized = jax.vmap(jax.vmap(self._normalize_desc))(desc)

        transitions = transitions.replace(
            desc=desc_normalized, desc_prime=desc_normalized
        )
        replay_buffer = replay_buffer.insert(transitions)

        # Initial training state
        key, subkey = jax.random.split(key)
        emitter_state = DCRLEmitterState(
            critic_params=critic_params,
            critic_opt_state=critic_opt_state,
            actor_params=actor_params,
            actor_opt_state=actor_opt_state,
            target_critic_params=target_critic_params,
            target_actor_params=target_actor_params,
            replay_buffer=replay_buffer,
            key=subkey,
            steps=jnp.array(0),
        )

        return emitter_state, key

    @partial(jax.jit, static_argnames=("self",))
    def _similarity(self, descs_1: Descriptor, descs_2: Descriptor) -> jnp.array:
        """Compute the similarity between two batches of descriptors.
        Args:
            descs_1: batch of descriptors.
            descs_2: batch of descriptors.
        Returns:
            batch of similarity measures.
        """
        return jnp.exp(
            -jnp.linalg.norm(descs_1 - descs_2, axis=-1) / self._config.lengthscale
        )

    @partial(jax.jit, static_argnames=("self",))
    def _normalize_desc(self, desc: Descriptor) -> Descriptor:
        return (
            2
            * (desc - self._env.behavior_descriptor_limits[0])
            / (
                self._env.behavior_descriptor_limits[1]
                - self._env.behavior_descriptor_limits[0]
            )
            - 1
        )

    @partial(jax.jit, static_argnames=("self",))
    def _unnormalize_desc(self, desc_normalized: Descriptor) -> Descriptor:
        return 0.5 * (
            self._env.behavior_descriptor_limits[1]
            - self._env.behavior_descriptor_limits[0]
        ) * desc_normalized + 0.5 * (
            self._env.behavior_descriptor_limits[1]
            + self._env.behavior_descriptor_limits[0]
        )

    @partial(jax.jit, static_argnames=("self",))
    def _compute_equivalent_kernel_bias_with_desc(
        self, actor_dc_params: Params, desc: Descriptor
    ) -> Tuple[Params, Params]:
        """
        Compute the equivalent bias of the first layer of the actor network
        given a descriptor.
        """
        # Extract kernel and bias of the first layer
        kernel = actor_dc_params["params"]["Dense_0"]["kernel"]
        bias = actor_dc_params["params"]["Dense_0"]["bias"]

        # Compute the equivalent bias
        equivalent_kernel = kernel[: -desc.shape[0], :]
        equivalent_bias = bias + jnp.dot(desc, kernel[-desc.shape[0] :])

        return equivalent_kernel, equivalent_bias

    @partial(jax.jit, static_argnames=("self",))
    def _compute_equivalent_params_with_desc(
        self, actor_dc_params: Params, desc: Descriptor
    ) -> Params:
        desc_normalized = self._normalize_desc(desc)
        (
            equivalent_kernel,
            equivalent_bias,
        ) = self._compute_equivalent_kernel_bias_with_desc(
            actor_dc_params, desc_normalized
        )
        actor_dc_params["params"]["Dense_0"]["kernel"] = equivalent_kernel
        actor_dc_params["params"]["Dense_0"]["bias"] = equivalent_bias
        return actor_dc_params

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: DCRLEmitterState,
        key: RNGKey,
    ) -> Tuple[Genotype, ExtraScores, RNGKey]:
        """Do a step of PG emission.

        Args:
            repertoire: the current repertoire of genotypes
            emitter_state: the state of the emitter used
            key: a random key

        Returns:
            A batch of offspring, the new emitter state and a new key.
        """
        # PG emitter
        parents_pg, descs_pg, key = repertoire.sample_with_descs(
            key, self._config.dcrl_batch_size
        )
        genotypes_pg = self.emit_pg(emitter_state, parents_pg, descs_pg)

        # Actor injection emitter
        _, descs_ai, key = repertoire.sample_with_descs(key, self._config.ai_batch_size)
        descs_ai = descs_ai.reshape(
            descs_ai.shape[0], self._env.behavior_descriptor_length
        )
        genotypes_ai = self.emit_ai(emitter_state, descs_ai)

        # Concatenate PG and AI genotypes
        genotypes = jax.tree_util.tree_map(
            lambda x1, x2: jnp.concatenate((x1, x2), axis=0), genotypes_pg, genotypes_ai
        )

        return (
            genotypes,
            {"desc_prime": jnp.concatenate([descs_pg, descs_ai], axis=0)},
            key,
        )

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit_pg(
        self,
        emitter_state: DCRLEmitterState,
        parents: Genotype,
        descs: Descriptor,
    ) -> Genotype:
        """Emit the offsprings generated through pg mutation.

        Args:
            emitter_state: current emitter state, contains critic and
                replay buffer.
            parents: the parents selected to be applied gradients in order
                to mutate towards better performance.
            descs: the descriptors of the parents.

        Returns:
            A new set of offsprings.
        """
        mutation_fn = partial(
            self._mutation_function_pg,
            emitter_state=emitter_state,
        )
        offsprings = jax.vmap(mutation_fn)(parents, descs)

        return offsprings

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit_ai(self, emitter_state: DCRLEmitterState, descs: Descriptor) -> Genotype:
        """Emit the offsprings generated through pg mutation.

        Args:
            emitter_state: current emitter state, contains critic and
                replay buffer.
            parents: the parents selected to be applied gradients in order
                to mutate towards better performance.
            descs: the descriptors of the parents.

        Returns:
            A new set of offsprings.
        """
        offsprings = jax.vmap(
            self._compute_equivalent_params_with_desc, in_axes=(None, 0)
        )(emitter_state.actor_params, descs)

        return offsprings

    @partial(jax.jit, static_argnames=("self",))
    def emit_actor(self, emitter_state: DCRLEmitterState) -> Genotype:
        """Emit the greedy actor.

        Simply needs to be retrieved from the emitter state.

        Args:
            emitter_state: the current emitter state, it stores the
                greedy actor.

        Returns:
            The parameters of the actor.
        """
        return emitter_state.actor_params

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: DCRLEmitterState,
        repertoire: Repertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> DCRLEmitterState:
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
        assert "transitions" in extra_scores.keys(), "Missing transitions or wrong key"
        transitions = extra_scores["transitions"]
        episode_length = transitions.obs.shape[1]

        desc_prime = jnp.concatenate(
            [
                extra_scores["desc_prime"],
                descriptors[
                    self._config.dcrl_batch_size + self._config.ai_batch_size :
                ],
            ],
            axis=0,
        )
        desc_prime = jnp.repeat(desc_prime[:, jnp.newaxis, :], episode_length, axis=1)
        desc = jnp.repeat(descriptors[:, jnp.newaxis, :], episode_length, axis=1)

        desc_prime_normalized = jax.vmap(jax.vmap(self._normalize_desc))(desc_prime)
        desc_normalized = jax.vmap(jax.vmap(self._normalize_desc))(desc)
        transitions = transitions.replace(
            desc=desc_normalized, desc_prime=desc_prime_normalized
        )

        # Add transitions to replay buffer
        replay_buffer = emitter_state.replay_buffer.insert(transitions)
        emitter_state = emitter_state.replace(replay_buffer=replay_buffer)

        # sample transitions from the replay buffer
        key, subkey = jax.random.split(emitter_state.key)
        transitions, key = replay_buffer.sample(
            subkey, self._config.num_critic_training_steps * self._config.batch_size
        )
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(
                x,
                (
                    self._config.num_critic_training_steps,
                    self._config.batch_size,
                    *x.shape[1:],
                ),
            ),
            transitions,
        )
        transitions = transitions.replace(
            rewards=self._similarity(transitions.desc, transitions.desc_prime)
            * transitions.rewards
        )
        emitter_state = emitter_state.replace(key=key)

        def scan_train_critics(
            carry: DCRLEmitterState,
            transitions: DCRLTransition,
        ) -> Tuple[DCRLEmitterState, Any]:
            emitter_state = carry
            new_emitter_state = self._train_critics(emitter_state, transitions)
            return new_emitter_state, ()

        # Train critics and greedy actor
        emitter_state, _ = jax.lax.scan(
            scan_train_critics,
            emitter_state,
            transitions,
            length=self._config.num_critic_training_steps,
        )

        return emitter_state  # type: ignore

    @partial(jax.jit, static_argnames=("self",))
    def _train_critics(
        self, emitter_state: DCRLEmitterState, transitions: DCRLTransition
    ) -> DCRLEmitterState:
        """Apply one gradient step to critics and to the greedy actor
        (contained in carry in training_state), then soft update target critics
        and target actor.

        Those updates are very similar to those made in TD3.

        Args:
            emitter_state: actual emitter state

        Returns:
            New emitter state where the critic and the greedy actor have been
            updated. Optimizer states have also been updated in the process.
        """
        # Update Critic
        (
            critic_opt_state,
            critic_params,
            target_critic_params,
            key,
        ) = self._update_critic(
            critic_params=emitter_state.critic_params,
            target_critic_params=emitter_state.target_critic_params,
            target_actor_params=emitter_state.target_actor_params,
            critic_opt_state=emitter_state.critic_opt_state,
            transitions=transitions,
            key=emitter_state.key,
        )

        # Update greedy actor
        (
            actor_opt_state,
            actor_params,
            target_actor_params,
        ) = jax.lax.cond(
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
            critic_opt_state=critic_opt_state,
            actor_params=actor_params,
            actor_opt_state=actor_opt_state,
            target_critic_params=target_critic_params,
            target_actor_params=target_actor_params,
            key=key,
            steps=emitter_state.steps + 1,
        )

        return new_emitter_state  # type: ignore

    @partial(jax.jit, static_argnames=("self",))
    def _update_critic(
        self,
        critic_params: Params,
        target_critic_params: Params,
        target_actor_params: Params,
        critic_opt_state: Params,
        transitions: DCRLTransition,
        key: RNGKey,
    ) -> Tuple[Params, Params, Params, RNGKey]:

        # compute loss and gradients
        key, subkey = jax.random.split(key)
        critic_gradient = jax.grad(self._critic_loss_fn)(
            critic_params,
            target_actor_params,
            target_critic_params,
            transitions,
            subkey,
        )
        critic_updates, critic_opt_state = self._critic_optimizer.update(
            critic_gradient, critic_opt_state
        )

        # update critic
        critic_params = optax.apply_updates(critic_params, critic_updates)

        # Soft update of target critic network
        target_critic_params = jax.tree_util.tree_map(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            target_critic_params,
            critic_params,
        )

        return critic_opt_state, critic_params, target_critic_params, key

    @partial(jax.jit, static_argnames=("self",))
    def _update_actor(
        self,
        actor_params: Params,
        actor_opt_state: optax.OptState,
        target_actor_params: Params,
        critic_params: Params,
        transitions: DCRLTransition,
    ) -> Tuple[optax.OptState, Params, Params]:

        # Update greedy actor
        policy_gradient = jax.grad(self._actor_loss_fn)(
            actor_params,
            critic_params,
            transitions,
        )
        (
            policy_updates,
            actor_opt_state,
        ) = self._actor_optimizer.update(policy_gradient, actor_opt_state)
        actor_params = optax.apply_updates(actor_params, policy_updates)

        # Soft update of target greedy actor
        target_actor_params = jax.tree_util.tree_map(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            target_actor_params,
            actor_params,
        )

        return (
            actor_opt_state,
            actor_params,
            target_actor_params,
        )

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def _mutation_function_pg(
        self,
        policy_params: Genotype,
        descs: Descriptor,
        emitter_state: DCRLEmitterState,
    ) -> Genotype:
        """Apply pg mutation to a policy via multiple steps of gradient descent.
        First, update the rewards to be diversity rewards, then apply the gradient
        steps.

        Args:
            policy_params: a policy, supposed to be a differentiable neural
                network.
            emitter_state: the current state of the emitter, containing among others,
                the replay buffer, the critic.

        Returns:
            The updated params of the neural network.
        """
        # Get transitions
        transitions, key = emitter_state.replay_buffer.sample(
            emitter_state.key,
            sample_size=self._config.num_pg_training_steps * self._config.batch_size,
        )
        descs_prime = jnp.tile(
            descs, (self._config.num_pg_training_steps * self._config.batch_size, 1)
        )
        descs_prime_normalized = jax.vmap(self._normalize_desc)(descs_prime)
        transitions = transitions.replace(
            rewards=self._similarity(transitions.desc, descs_prime_normalized)
            * transitions.rewards,
            desc_prime=descs_prime_normalized,
        )
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(
                x,
                (
                    self._config.num_pg_training_steps,
                    self._config.batch_size,
                    *x.shape[1:],
                ),
            ),
            transitions,
        )

        # Replace key
        emitter_state = emitter_state.replace(key=key)

        # Define new policy optimizer state
        policy_opt_state = self._policies_optimizer.init(policy_params)

        def scan_train_policy(
            carry: Tuple[DCRLEmitterState, Genotype, optax.OptState],
            transitions: DCRLTransition,
        ) -> Tuple[Tuple[DCRLEmitterState, Genotype, optax.OptState], Any]:
            emitter_state, policy_params, policy_opt_state = carry
            (
                new_emitter_state,
                new_policy_params,
                new_policy_opt_state,
            ) = self._train_policy(
                emitter_state,
                policy_params,
                policy_opt_state,
                transitions,
            )
            return (
                new_emitter_state,
                new_policy_params,
                new_policy_opt_state,
            ), ()

        (
            emitter_state,
            policy_params,
            policy_opt_state,
        ), _ = jax.lax.scan(
            scan_train_policy,
            (emitter_state, policy_params, policy_opt_state),
            transitions,
            length=self._config.num_pg_training_steps,
        )

        return policy_params

    @partial(jax.jit, static_argnames=("self",))
    def _train_policy(
        self,
        emitter_state: DCRLEmitterState,
        policy_params: Params,
        policy_opt_state: optax.OptState,
        transitions: DCRLTransition,
    ) -> Tuple[DCRLEmitterState, Params, optax.OptState]:
        """Apply one gradient step to a policy (called policy_params).

        Args:
            emitter_state: current state of the emitter.
            policy_params: parameters corresponding to the weights and bias of
                the neural network that defines the policy.

        Returns:
            The new emitter state and new params of the NN.
        """
        # update policy
        policy_opt_state, policy_params = self._update_policy(
            critic_params=emitter_state.critic_params,
            policy_opt_state=policy_opt_state,
            policy_params=policy_params,
            transitions=transitions,
        )

        return emitter_state, policy_params, policy_opt_state

    @partial(jax.jit, static_argnames=("self",))
    def _update_policy(
        self,
        critic_params: Params,
        policy_opt_state: optax.OptState,
        policy_params: Params,
        transitions: DCRLTransition,
    ) -> Tuple[optax.OptState, Params]:

        # compute loss
        policy_gradient = jax.grad(self._policy_loss_fn)(
            policy_params,
            critic_params,
            transitions,
        )
        # Compute gradient and update policies
        (
            policy_updates,
            policy_opt_state,
        ) = self._policies_optimizer.update(policy_gradient, policy_opt_state)
        policy_params = optax.apply_updates(policy_params, policy_updates)

        return policy_opt_state, policy_params
