""" Implements the PGA-ME algorithm in jax for brax environments, based on:
https://hal.archives-ouvertes.fr/hal-03135723v2/file/PGA_MAP_Elites_GECCO.pdf"""
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import optax
from jax import numpy as jnp
from jax.tree_util import tree_map

from qdax.algorithms.map_elites import MapElitesRepertoire
from qdax.brax_envs.utils_wrappers import QDEnv
from qdax.types import Genotype, Params, RNGKey
from qdax.utils.buffers import FlatBuffer
from qdax.utils.mdp_utils import QDTransition, Transition
from qdax.utils.networks import QModule
from qdax.utils.td3_utils import make_td3_loss_fn


@dataclass
class PGAMEConfig:
    """Configuration for PGAME Algorithm"""

    proportion_mutation_ga: float = 0.5

    # TD3 params
    replay_buffer_size: int = 1000000
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-3
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    transitions_batch_size: int = 256
    expl_noise: float = 0.1
    tau: float = 0.005
    nb_critic_training_steps: int = 300
    nb_pg_training_steps: int = 10


@flax.struct.dataclass
class PGEmitterState:
    """Contains training state for the learner."""

    q_params: Params
    q_optimizer_state: optax.OptState
    greedy_policy_params: Params
    greedy_policy_opt_state: optax.OptState
    controllers_optimizer_state: optax.OptState
    target_q_params: Params
    target_greedy_policy_params: Params
    replay_buffer: FlatBuffer
    random_key: RNGKey
    steps: jnp.ndarray


class PGEmitter:
    """
    A policy gradient emitter used to implement the Policy Gradient Assisted MAP-Elites
    (PGA-Map-Elites) algorithm.
    """

    def __init__(
        self,
        config: PGAMEConfig,
        policy_network: nn.Module,
        env: QDEnv,
        unroll_fn: Callable[[Params], QDTransition],
        crossover_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
    ):
        self._config = config
        self._env = env
        self._crossover_fn = crossover_fn
        self._unroll_fn = unroll_fn
        self._policy_network = policy_network

        # Placeholder for the losses and optimizers
        self._policy_loss_fn: Optional[
            Callable[[Params, Params, Transition], jnp.ndarray]
        ] = None
        self._critic_loss_fn: Optional[
            Callable[[Params, Params, Params, Transition, RNGKey], jnp.ndarray]
        ] = None
        self._greedy_policy_optimizer: Optional[optax.GradientTransformation] = None
        self._critic_optimizer: Optional[optax.GradientTransformation] = None
        self._controllers_optimizer: Optional[optax.GradientTransformation] = None

    def _init_networks_and_params(self, init_genotypes: Genotype, random_key: RNGKey):
        """Create networks and params used in PGAME."""

        observation_size = self._env.observation_size
        action_size = self._env.action_size

        # Init Critics and greedy policy (named according to paper)
        random_key, subkey = jax.random.split(random_key)
        critic_network = QModule(
            n_critics=2, hidden_layer_sizes=self._config.critic_hidden_layer_size
        )
        fake_obs = jnp.zeros(shape=(observation_size,))
        fake_action = jnp.zeros(shape=(action_size,))
        critic_params = critic_network.init(subkey, obs=fake_obs, actions=fake_action)
        target_critic_params = tree_map(lambda x: jnp.asarray(x.copy()), critic_params)

        greedy_policy_params = tree_map(
            lambda x: jnp.asarray(x[0].copy()), init_genotypes
        )
        target_greedy_policy_params = tree_map(
            lambda x: jnp.asarray(x[0].copy()), init_genotypes
        )

        return (
            self._policy_network,
            critic_network,
            init_genotypes,
            greedy_policy_params,
            target_greedy_policy_params,
            critic_params,
            target_critic_params,
            random_key,
        )

    def _init_loss_and_optimizers(
        self,
        policy_network: nn.Module,
        q_network: nn.Module,
        critic_params: Params,
        policy_params: Params,
    ) -> Tuple[optax.OptState, optax.OptState, optax.OptState]:
        """
        Set the loss functions and optimizers.
        """

        policy_loss_fn, critic_loss_fn = make_td3_loss_fn(
            policy_network=policy_network,
            q_network=q_network,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            noise_clip=self._config.noise_clip,
            policy_noise=self._config.policy_noise,
        )
        self._policy_loss_fn = policy_loss_fn
        self._critic_loss_fn = critic_loss_fn

        # Init optimizers
        self._greedy_policy_optimizer = optax.adam(
            learning_rate=self._config.policy_learning_rate
        )
        self._critic_optimizer = optax.adam(
            learning_rate=self._config.critic_learning_rate
        )
        self._controllers_optimizer = optax.adam(
            learning_rate=self._config.policy_learning_rate
        )

        # Prepare init states and return them to the user
        critic_optimizer_state = self._critic_optimizer.init(critic_params)
        greedy_optimizer_state = self._greedy_policy_optimizer.init(policy_params)
        controllers_optimizer_state = self._controllers_optimizer.init(policy_params)

        return (
            critic_optimizer_state,
            greedy_optimizer_state,
            controllers_optimizer_state,
        )

    def init_fn(self, init_genotypes: Genotype, random_key: RNGKey) -> PGEmitterState:
        """
        Initializes the emitter state.
        """

        observation_size = self._env.observation_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_length

        # Initialise critic, greedy and population
        (
            policy_network,
            critic_network,
            init_genotypes,
            greedy_policy_params,
            target_greedy_policy_params,
            critic_params,
            target_critic_params,
            random_key,
        ) = self._init_networks_and_params(
            init_genotypes=init_genotypes, random_key=random_key
        )

        # Set up the losses and optimizers - return the opt states
        (
            critic_optimizer_state,
            greedy_policy_opt_state,
            controllers_optimizer_state,
        ) = self._init_loss_and_optimizers(
            policy_network=policy_network,
            q_network=critic_network,
            critic_params=critic_params,
            policy_params=greedy_policy_params,
        )

        # Initialize replay buffer
        dummy_transition = QDTransition.init_dummy(
            observation_dim=observation_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )

        replay_buffer = FlatBuffer.init(
            buffer_size=self._config.replay_buffer_size, transition=dummy_transition
        )

        # Initial training state
        training_state = PGEmitterState(
            q_params=critic_params,
            q_optimizer_state=critic_optimizer_state,
            greedy_policy_params=greedy_policy_params,
            greedy_policy_opt_state=greedy_policy_opt_state,
            controllers_optimizer_state=controllers_optimizer_state,
            target_q_params=target_critic_params,
            target_greedy_policy_params=target_greedy_policy_params,
            random_key=random_key,
            steps=jnp.array(0),
            replay_buffer=replay_buffer,
        )

        return training_state

    @partial(
        jax.jit,
        static_argnames=("self", "batch_size"),
    )
    def emitter_fn(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: PGEmitterState,
        random_key: RNGKey,
        batch_size: int,
    ) -> Tuple[Genotype, PGEmitterState, RNGKey]:
        """
        Do a single PGA-ME iteration: train critics and greedy policy,
        make mutations (evo and pg), score solution, fill replay buffer and insert back
        in the MAP-Elites grid."""

        def scan_train_critics(carry, unused):
            emitter_state = carry
            new_emitter_state = self._train_critics(emitter_state)
            return new_emitter_state, ()

        # Train critics
        emitter_state, _ = jax.lax.scan(
            scan_train_critics,
            emitter_state,
            (),
            length=self._config.nb_critic_training_steps,
        )

        # Mutation evo
        mutation_ga_batch_size = int(self._config.proportion_mutation_ga * batch_size)
        x1, random_key = repertoire.sample(random_key, mutation_ga_batch_size)
        x2, random_key = repertoire.sample(random_key, mutation_ga_batch_size)
        x_mutation_ga, random_key = self._crossover_fn(x1, x2, random_key)

        # Mutation PG
        mutation_pg_batch_size = int(batch_size - mutation_ga_batch_size - 1)
        x1, random_key = repertoire.sample(random_key, mutation_pg_batch_size)
        mutation_fn = partial(
            self._mutation_function_pg,
            emitter_state=emitter_state,
        )
        x_mutation_pg = jax.vmap(mutation_fn)(x1)

        # Add dimension for concatenation
        greedy_policy_params = jax.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), emitter_state.greedy_policy_params
        )

        # Score all new solutions, fill replay buffer with evaluations
        # and add solutions in grid
        genotypes = jax.tree_multimap(
            lambda x, y, z: jnp.concatenate([x, y, z], axis=0),
            x_mutation_ga,
            x_mutation_pg,
            greedy_policy_params,
        )

        transitions = self._unroll_fn(genotypes)
        replay_buffer = emitter_state.replay_buffer
        replay_buffer = replay_buffer.insert(transitions)

        emitter_state = emitter_state.replace(replay_buffer=replay_buffer)

        return genotypes, emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def _train_critics(self, emitter_state: PGEmitterState) -> PGEmitterState:
        """Apply one gradient step to critics and to the greedy policy
        (contained in carry in training_state), then soft update target critics
        and target greedy policy."""

        # Sample a batch of transitions in the buffer
        key = emitter_state.random_key
        key, subkey = jax.random.split(key)
        replay_buffer = emitter_state.replay_buffer
        samples = replay_buffer.sample(
            subkey, sample_size=self._config.transitions_batch_size
        )

        # Update Critic
        key, subkey = jax.random.split(key)
        q_loss, q_gradient = jax.value_and_grad(self._critic_loss_fn)(
            emitter_state.q_params,
            emitter_state.target_greedy_policy_params,
            emitter_state.target_q_params,
            samples,
            subkey,
        )
        q_updates, q_optimizer_state = self._critic_optimizer.update(
            q_gradient, emitter_state.q_optimizer_state
        )
        q_params = optax.apply_updates(emitter_state.q_params, q_updates)
        # Soft update of target q network
        target_q_params = jax.tree_multimap(
            lambda x1, x2: (1.0 - self._config.tau) * x1 + self._config.tau * x2,
            emitter_state.target_q_params,
            q_params,
        )

        # Update greedy policy
        key, subkey = jax.random.split(key)
        policy_loss, policy_gradient = jax.value_and_grad(self._policy_loss_fn)(
            emitter_state.greedy_policy_params,
            emitter_state.q_params,
            samples,
        )
        (
            policy_updates,
            policy_optimizer_state,
        ) = self._greedy_policy_optimizer.update(
            policy_gradient, emitter_state.greedy_policy_opt_state
        )
        greedy_policy_params = optax.apply_updates(
            emitter_state.greedy_policy_params, policy_updates
        )
        # Soft update of target greedy policy
        target_greedy_policy_params = jax.tree_multimap(
            lambda x1, x2: (1.0 - self._config.tau) * x1 + self._config.tau * x2,
            emitter_state.target_greedy_policy_params,
            greedy_policy_params,
        )

        # Create new training state
        new_state = PGEmitterState(
            q_params=q_params,
            q_optimizer_state=q_optimizer_state,
            greedy_policy_params=greedy_policy_params,
            greedy_policy_opt_state=policy_optimizer_state,
            controllers_optimizer_state=emitter_state.controllers_optimizer_state,
            target_q_params=target_q_params,
            target_greedy_policy_params=target_greedy_policy_params,
            random_key=key,
            steps=emitter_state.steps + 1,
            replay_buffer=replay_buffer,
        )

        return new_state

    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_pg(
        self,
        controller_params: Genotype,
        emitter_state: PGEmitterState,
    ) -> Params:
        """Apply pg mutation to a policy via multiple steps of gradient descent"""

        def scan_train_controller(carry, unused):
            emitter_state, controller_params = carry
            (
                new_emitter_state,
                new_controller_params,
            ) = self._train_controller(emitter_state, controller_params)
            return (new_emitter_state, new_controller_params), ()

        (emitter_state, controller_params), _ = jax.lax.scan(
            scan_train_controller,
            (emitter_state, controller_params),
            (),
            length=self._config.nb_pg_training_steps,
        )

        return controller_params

    @partial(jax.jit, static_argnames=("self",))
    def _train_controller(
        self,
        emitter_state: PGEmitterState,
        controller_params: Params,
    ) -> Tuple[PGEmitterState, Params]:
        """
        Apply one gradient step to a policy (called controllers_params)
        """

        # Sample a batch of transitions in the buffer
        key = emitter_state.random_key
        key, subkey = jax.random.split(key)
        replay_buffer = emitter_state.replay_buffer
        samples = replay_buffer.sample(
            subkey, sample_size=self._config.transitions_batch_size
        )
        policy_loss, policy_gradient = jax.value_and_grad(self._policy_loss_fn)(
            controller_params,
            emitter_state.q_params,
            samples,
        )
        # Compute gradient and update policies
        (policy_updates, policy_optimizer_state,) = self._controllers_optimizer.update(
            policy_gradient, emitter_state.controllers_optimizer_state
        )
        controller_params = optax.apply_updates(controller_params, policy_updates)

        # Create new training state
        new_emitter_state = PGEmitterState(
            q_params=emitter_state.q_params,
            q_optimizer_state=emitter_state.q_optimizer_state,
            greedy_policy_params=emitter_state.greedy_policy_params,
            greedy_policy_opt_state=emitter_state.greedy_policy_opt_state,
            controllers_optimizer_state=policy_optimizer_state,
            target_q_params=emitter_state.target_q_params,
            target_greedy_policy_params=emitter_state.target_greedy_policy_params,
            random_key=key,
            steps=emitter_state.steps,
            replay_buffer=replay_buffer,
        )

        return new_emitter_state, controller_params
