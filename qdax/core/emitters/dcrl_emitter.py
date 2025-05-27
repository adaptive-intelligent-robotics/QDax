"""Implements the DCRL Emitter from DCRL-MAP-Elites algorithm
in JAX for Brax environments.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.emitters.repertoire_selectors.selector import Selector
from qdax.core.neuroevolution.buffers.buffer import DCRLTransition, ReplayBuffer
from qdax.core.neuroevolution.losses.td3_loss import make_td3_loss_dc_fn
from qdax.core.neuroevolution.networks.networks import QModuleDC
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey
from qdax.tasks.brax.v1.envs.base_env import QDEnv


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
        selector: Optional[Selector] = None,
    ) -> None:
        self._config = config
        self._env = env
        self._selector = selector
        self._actor_network = actor_network
        self._actor_critic_iterations = int(
            config.num_critic_training_steps / config.policy_delay
        )  # actor and critic training are packed into a single function

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
        """Whether to use all data or not when used along other emitters"""
        return True

    def init(
        self,
        key: RNGKey,
        repertoire: Repertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> DCRLEmitterState:
        """Initializes the emitter state.

        Args:
            genotypes: The initial population.
            key: A random key.

        Returns:
            The initial state of the PGAMEEmitter
        """

        observation_size = jax.tree.leaves(genotypes)[1].shape[1]
        descriptor_size = self._env.descriptor_length
        action_size = self._env.action_size

        # Initialise critic, greedy actor and population
        key, subkey = jax.random.split(key)
        fake_obs = jnp.zeros(shape=(observation_size,))
        fake_desc = jnp.zeros(shape=(descriptor_size,))
        fake_action = jnp.zeros(shape=(action_size,))

        critic_params = self._critic_network.init(
            subkey, obs=fake_obs, actions=fake_action, desc=fake_desc
        )
        target_critic_params = jax.tree.map(lambda x: x, critic_params)

        key, subkey = jax.random.split(key)
        actor_params = self._actor_network.init(subkey, obs=fake_obs, desc=fake_desc)
        target_actor_params = jax.tree.map(lambda x: x, actor_params)

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

        desc = jnp.repeat(descriptors[:, None, :], episode_length, axis=1)
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
        )

        return emitter_state

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

    def _normalize_desc(self, desc: Descriptor) -> Descriptor:
        return (
            2
            * (desc - self._env.descriptor_limits[0])
            / (self._env.descriptor_limits[1] - self._env.descriptor_limits[0])
            - 1
        )

    def _unnormalize_desc(self, desc_normalized: Descriptor) -> Descriptor:
        return 0.5 * (
            self._env.descriptor_limits[1] - self._env.descriptor_limits[0]
        ) * desc_normalized + 0.5 * (
            self._env.descriptor_limits[1] + self._env.descriptor_limits[0]
        )

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

    def emit(  # type: ignore
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: DCRLEmitterState,
        key: RNGKey,
    ) -> Tuple[Genotype, ExtraScores]:
        """Do a step of policy-gradient and actor-injection emission.

        Args:
            repertoire: the current repertoire of genotypes
            emitter_state: the state of the emitter used
            key: a random key

        Returns:
            A batch of offspring with intended descs as extra information
        """
        # PG emitter
        subkey1, subkey2 = jax.random.split(key)
        sub_repertoire = repertoire.select(
            subkey1, self._config.dcrl_batch_size, selector=self._selector
        )
        parents_pg = sub_repertoire.genotypes
        descs_pg = sub_repertoire.descriptors
        genotypes_pg = self.emit_pg(emitter_state, parents_pg, descs_pg)

        # Actor injection emitter
        descs_ai = repertoire.select(
            subkey2, self._config.ai_batch_size, selector=self._selector
        ).descriptors
        descs_ai = descs_ai.reshape(descs_ai.shape[0], self._env.descriptor_length)
        genotypes_ai = self.emit_ai(emitter_state, descs_ai)

        # Concatenate PG and AI genotypes
        genotypes = jax.tree.map(
            lambda x, y: jnp.concatenate((x, y), axis=0), genotypes_pg, genotypes_ai
        )

        return (
            genotypes,
            {"desc_prime": jnp.concatenate([descs_pg, descs_ai], axis=0)},
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
        # normalize the intended descriptors and prepare them for concatenation
        descs_prime_normalized = jnp.tile(
            jax.vmap(self._normalize_desc)(descs), self._config.batch_size
        ).reshape(self._config.dcrl_batch_size, self._config.batch_size, -1)

        # create a batch of policy optimizer states
        policy_opt_states = jax.vmap(self._policies_optimizer.init)(parents)

        # prepare the batched policy update function with vmapping
        batched_policy_update_fn = jax.vmap(
            partial(self._update_policy, critic_params=emitter_state.critic_params),
            in_axes=(0, 0, 0, None),
        )

        def scan_update_policies(
            carry: Tuple[Params, optax.OptState, jnp.ndarray, RNGKey],
            _: None,
        ) -> Tuple[Tuple[Params, optax.OptState, jnp.ndarray, RNGKey], Any]:

            (policy_params, policy_opt_state, desc_prime, key) = carry
            key, subkey = jax.random.split(key)

            # sample a mini-batch of data from the replay-buffer
            transitions = emitter_state.replay_buffer.sample(
                subkey, self._config.batch_size
            )
            (
                new_policy_params,
                new_policy_opt_states,
            ) = batched_policy_update_fn(
                policy_params, policy_opt_state, desc_prime, transitions
            )
            return (new_policy_params, new_policy_opt_states, desc_prime, key), ()

        (
            final_policy_params,
            policy_opt_state,
            desc_prime,
            key,
        ), _ = jax.lax.scan(
            scan_update_policies,
            (parents, policy_opt_states, descs_prime_normalized, emitter_state.key),
            length=self._config.num_pg_training_steps,
        )

        return final_policy_params

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

    def state_update(  # type: ignore
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
        transitions: DCRLTransition,
    ) -> Tuple[Tuple[Params, Params, optax.OptState, Params, RNGKey], Any]:
        """A scan-ready function to update the critic network parameters
        with one gradient step.

        Args:
            carry: packed carry containing critic parameters, target critic
                parameters, critic optimiser state, target actor parameters,
                and the random key.
            transitions: a mini-batch of DCRLtransitions for gradient computation

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

        # compute critic gradients
        key, subkey = jax.random.split(key)
        critic_gradient = jax.grad(self._critic_loss_fn)(
            critic_params,
            target_actor_params,
            target_critic_params,
            transitions,
            subkey,
        )
        critic_updates, new_critic_opt_state = self._critic_optimizer.update(
            critic_gradient, critic_opt_state
        )

        # update critic
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
        transitions: DCRLTransition,
    ) -> Tuple[optax.OptState, Params, Params]:
        """Function to update the DC-actor with one Policy-Gradient step.

        Args:
            actor_params: neural network parameters of the DC-actor.
            actor_opt_state: optimiser state of the actor.
            target_actor_params: target actor parameters (used in TD3 to
                smooth the actor-critic learning).
            critic_params: the parameters of the critic networks which plays
                as a differentiable target.
            transitions: a mini-batch of DCRLtransitions.

        Returns:
            new_actor_params: new actor parameters after taking the PG step
            new_target_actor_params: new target actor parameters (soft-tau update)
            new_actor_opt_state: updated optimiser state
        """

        # Update greedy actor
        policy_gradient = jax.grad(self._actor_loss_fn)(
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
        self, carry: DCRLEmitterState, _: None
    ) -> Tuple[DCRLEmitterState, Tuple]:
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
            subkey, self._config.batch_size * (self._config.policy_delay + 1)
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

        # rescale rewards by descriptor similarity
        transitions = transitions.replace(
            rewards=self._similarity(transitions.desc, transitions.desc_prime)
            * transitions.rewards
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
                emitter_state.critic_opt_state,
                emitter_state.target_actor_params,
                key,
            ),
            critic_data,
        )

        # train actor with one gradient step
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
            critic_opt_state=new_critic_opt_state,
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
        desc_prime: Descriptor,
        transitions: DCRLTransition,
        critic_params: Params,
    ) -> Tuple[Params, optax.OptState]:
        """Perform one step of PG update on the off-spring policy.
        This function is vmapped to mutate the entire batch of off-springs
        in parallel.

        Args:
            policy_params: the parameters of the policy.
            policy_opt_state: the optimiser state of the policy.
            desc_prime: the normalised intended descriptor of the policy.
            transitions: a mini-batch of transitions for gradient computation
            critic_params: the parameters of the critic networks serving as
                a differentiable target. This is fixed in each iteration.

        Returns:
            new_policy_params: new policy parameters
            new_policy_opt_state: updated optimiser state
        """

        # Compute gradient
        policy_gradient = jax.grad(self._policy_loss_fn)(
            policy_params,
            critic_params,
            desc_prime,
            transitions,
        )
        # Update the policy
        (
            policy_updates,
            new_policy_opt_state,
        ) = self._policies_optimizer.update(policy_gradient, policy_opt_state)
        new_policy_params = optax.apply_updates(policy_params, policy_updates)

        return new_policy_params, new_policy_opt_state
