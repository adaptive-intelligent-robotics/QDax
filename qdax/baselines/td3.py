""" Implements the TD3 algorithm in jax for brax environments, based on:
https://arxiv.org/pdf/1802.09477.pdf"""

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import optax
from brax.envs import Env
from brax.envs import State as EnvState
from jax import numpy as jnp

from qdax.core.neuroevolution.buffers.buffer import (
    QDTransition,
    ReplayBuffer,
    Transition,
)
from qdax.core.neuroevolution.losses.td3_loss import (
    td3_critic_loss_fn,
    td3_policy_loss_fn,
)
from qdax.core.neuroevolution.mdp_utils import TrainingState, get_first_episode
from qdax.core.neuroevolution.networks.td3_networks import make_td3_networks
from qdax.core.neuroevolution.sac_td3_utils import generate_unroll
from qdax.custom_types import (
    Action,
    Descriptor,
    Mask,
    Metrics,
    Observation,
    Params,
    Reward,
    RNGKey,
)


class TD3TrainingState(TrainingState):
    """Contains training state for the learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    critic_optimizer_state: optax.OptState
    critic_params: Params
    target_critic_params: Params
    target_policy_params: Params
    key: RNGKey
    steps: jnp.ndarray


@dataclass
class TD3Config:
    """Configuration for the TD3 algorithm"""

    episode_length: int = 1000
    batch_size: int = 256
    policy_delay: int = 2
    soft_tau_update: float = 0.005
    expl_noise: float = 0.1
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    policy_hidden_layer_size: Tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4
    policy_learning_rate: float = 3e-4
    discount: float = 0.99
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    reward_scaling: float = 1.0


class TD3:
    """
    A collection of functions that define the Twin Delayed Deep Deterministic Policy
    Gradient agent (TD3), ref: https://arxiv.org/pdf/1802.09477.pdf
    """

    def __init__(self, config: TD3Config, action_size: int):
        self._config = config

        (
            self._policy,
            self._critic,
        ) = make_td3_networks(
            action_size=action_size,
            critic_hidden_layer_sizes=self._config.critic_hidden_layer_size,
            policy_hidden_layer_sizes=self._config.policy_hidden_layer_size,
        )

    def init(
        self, key: RNGKey, action_size: int, observation_size: int
    ) -> TD3TrainingState:
        """Initialise the training state of the TD3 algorithm, through creation
        of optimizer states and params.

        Args:
            key: a random key used for random operations.
            action_size: the size of the action array needed to interact with the
                environment.
            observation_size: the size of the observation array retrieved from the
                environment.

        Returns:
            the initial training state.
        """

        # Initialize critics and policy params
        fake_obs = jnp.zeros(shape=(observation_size,))
        fake_action = jnp.zeros(shape=(action_size,))
        key, subkey_1, subkey_2 = jax.random.split(key, num=3)
        critic_params = self._critic.init(subkey_1, obs=fake_obs, actions=fake_action)
        policy_params = self._policy.init(subkey_2, fake_obs)

        # Initialize target networks
        target_critic_params = jax.tree.map(
            lambda x: jnp.asarray(x.copy()), critic_params
        )
        target_policy_params = jax.tree.map(
            lambda x: jnp.asarray(x.copy()), policy_params
        )

        # Create and initialize optimizers
        critic_optimizer_state = optax.adam(learning_rate=1.0).init(critic_params)
        policy_optimizer_state = optax.adam(learning_rate=1.0).init(policy_params)

        # Initial training state
        training_state = TD3TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            target_policy_params=target_policy_params,
            target_critic_params=target_critic_params,
            key=key,
            steps=jnp.array(0),
        )

        return training_state

    def select_action(
        self,
        obs: Observation,
        policy_params: Params,
        key: RNGKey,
        expl_noise: float,
        deterministic: bool = False,
    ) -> Action:
        """Selects an action according to TD3 policy. The action can be deterministic
        or stochastic by adding exploration noise.

        Args:
            obs: agent observation(s)
            policy_params: parameters of the agent's policy
            key: jax random key
            expl_noise: exploration noise
            deterministic: whether to select action in a deterministic way.
                Defaults to False.

        Returns:
            an action and an updated training state.
        """

        actions = self._policy.apply(policy_params, obs)
        if not deterministic:
            noise = jax.random.normal(key, actions.shape) * expl_noise
            actions = actions + noise
            actions = jnp.clip(actions, -1.0, 1.0)
        return actions

    def play_step_fn(
        self,
        env_state: EnvState,
        training_state: TD3TrainingState,
        env: Env,
        deterministic: bool = False,
    ) -> Tuple[EnvState, TD3TrainingState, Transition]:
        """Plays a step in the environment. Selects an action according to TD3 rule and
        performs the environment step.

        Args:
            env_state: the current environment state
            training_state: the SAC training state
            env: the environment
            deterministic: whether to select action in a deterministic way.
                Defaults to False.

        Returns:
            the new environment state
            the new TD3 training state
            the played transition
        """
        key = training_state.key

        key, subkey = jax.random.split(key)
        actions = self.select_action(
            obs=env_state.obs,
            policy_params=training_state.policy_params,
            key=subkey,
            expl_noise=self._config.expl_noise,
            deterministic=deterministic,
        )
        training_state = training_state.replace(
            key=key,
        )
        next_env_state = env.step(env_state, actions)
        transition = Transition(
            obs=env_state.obs,
            next_obs=next_env_state.obs,
            rewards=next_env_state.reward,
            dones=next_env_state.done,
            truncations=next_env_state.info["truncation"],
            actions=actions,
        )
        return next_env_state, training_state, transition

    def play_qd_step_fn(
        self,
        env_state: EnvState,
        training_state: TD3TrainingState,
        env: Env,
        deterministic: bool = False,
    ) -> Tuple[EnvState, TD3TrainingState, QDTransition]:
        """Plays a step in the environment. Selects an action according to TD3 rule and
        performs the environment step.

        Args:
            env_state: the current environment state
            training_state: the TD3 training state
            env: the environment
            deterministic: the whether to select action in a deterministic way.
                Defaults to False.

        Returns:
            the new environment state
            the new TD3 training state
            the played transition
        """
        next_env_state, training_state, transition = self.play_step_fn(
            env_state, training_state, env, deterministic
        )
        actions = transition.actions

        truncations = next_env_state.info["truncation"]
        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_env_state.obs,
            rewards=next_env_state.reward,
            dones=next_env_state.done,
            actions=actions,
            truncations=truncations,
            state_desc=env_state.info["state_descriptor"],
            next_state_desc=next_env_state.info["state_descriptor"],
        )

        return (
            next_env_state,
            training_state,
            transition,
        )

    def eval_policy_fn(
        self,
        training_state: TD3TrainingState,
        eval_env_first_state: EnvState,
        play_step_fn: Callable[
            [EnvState, Params],
            Tuple[EnvState, Params, Transition],
        ],
    ) -> Tuple[Reward, Reward]:
        """Evaluates the agent's policy over an entire episode, across all batched
        environments.

        Args:
            training_state: TD3 training state.
            eval_env_first_state: the first state of the environment.
            play_step_fn: function defining how to play a step in the env.

        Returns:
            true return averaged over batch dimension, shape: (1,)
            true return per env, shape: (env_batch_size,)
        """
        # TODO: this generate unroll shouldn't take a random key
        state, training_state, transitions = generate_unroll(
            init_state=eval_env_first_state,
            training_state=training_state,
            episode_length=self._config.episode_length,
            play_step_fn=play_step_fn,
        )

        transitions = get_first_episode(transitions)

        true_returns = jnp.nansum(transitions.rewards, axis=0)
        true_return = jnp.mean(true_returns, axis=-1)

        return true_return, true_returns

    def eval_qd_policy_fn(
        self,
        training_state: TD3TrainingState,
        eval_env_first_state: EnvState,
        play_step_fn: Callable[
            [EnvState, Params],
            Tuple[EnvState, TD3TrainingState, QDTransition],
        ],
        descriptor_extraction_fn: Callable[[QDTransition, Mask], Descriptor],
    ) -> Tuple[Reward, Descriptor, Reward, Descriptor]:
        """Evaluates the agent's policy over an entire episode, across all batched
        environments for QD environments. Averaged descriptors are returned as well.


        Args:
            training_state: the SAC training state
            eval_env_first_state: the initial state for evaluation
            play_step_fn: the play_step function used to collect the evaluation episode

        Returns:
            the true return averaged over batch dimension, shape: (1,)
            the descriptor averaged over batch dimension, shape: (num_descriptors,)
            the true return per environment, shape: (env_batch_size,)
            the descriptor per environment, shape: (env_batch_size, num_descriptors)

        """

        state, training_state, transitions = generate_unroll(
            init_state=eval_env_first_state,
            training_state=training_state,
            episode_length=self._config.episode_length,
            play_step_fn=play_step_fn,
        )
        transitions = get_first_episode(transitions)
        true_returns = jnp.nansum(transitions.rewards, axis=0)
        true_return = jnp.mean(true_returns, axis=-1)

        transitions = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), transitions)
        masks = jnp.isnan(transitions.rewards)
        descriptors = descriptor_extraction_fn(transitions, masks)

        mean_descriptor = jnp.mean(descriptors, axis=0)
        return true_return, mean_descriptor, true_returns, descriptors

    def update(
        self,
        training_state: TD3TrainingState,
        replay_buffer: ReplayBuffer,
    ) -> Tuple[TD3TrainingState, ReplayBuffer, Metrics]:
        """Performs a single training step: updates policy params and critic params
        through gradient descent.

        Args:
            training_state: the current training state, containing the optimizer states
                and the params of the policy and critic.
            replay_buffer: the replay buffer, filled with transitions experienced in
                the environment.

        Returns:
            A new training state, the buffer with new transitions and metrics about the
            training process.
        """

        # Sample a batch of transitions in the buffer
        key = training_state.key

        key, subkey = jax.random.split(key)
        samples = replay_buffer.sample(subkey, sample_size=self._config.batch_size)

        # Update Critic
        key, subkey = jax.random.split(key)
        critic_loss, critic_gradient = jax.value_and_grad(td3_critic_loss_fn)(
            training_state.critic_params,
            target_policy_params=training_state.target_policy_params,
            target_critic_params=training_state.target_critic_params,
            policy_fn=self._policy.apply,
            critic_fn=self._critic.apply,
            policy_noise=self._config.policy_noise,
            noise_clip=self._config.noise_clip,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            transitions=samples,
            key=subkey,
        )
        critic_optimizer = optax.adam(learning_rate=self._config.critic_learning_rate)
        critic_updates, critic_optimizer_state = critic_optimizer.update(
            critic_gradient, training_state.critic_optimizer_state
        )
        critic_params = optax.apply_updates(
            training_state.critic_params, critic_updates
        )
        # Soft update of target critic network
        target_critic_params = jax.tree.map(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            training_state.target_critic_params,
            critic_params,
        )

        # Update policy
        policy_loss, policy_gradient = jax.value_and_grad(td3_policy_loss_fn)(
            training_state.policy_params,
            critic_params=training_state.critic_params,
            policy_fn=self._policy.apply,
            critic_fn=self._critic.apply,
            transitions=samples,
        )

        def update_policy_step() -> Tuple[Params, Params, optax.OptState]:
            policy_optimizer = optax.adam(
                learning_rate=self._config.policy_learning_rate
            )
            (
                policy_updates,
                policy_optimizer_state,
            ) = policy_optimizer.update(
                policy_gradient, training_state.policy_optimizer_state
            )
            policy_params = optax.apply_updates(
                training_state.policy_params, policy_updates
            )
            # Soft update of target policy
            target_policy_params = jax.tree.map(
                lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
                + self._config.soft_tau_update * x2,
                training_state.target_policy_params,
                policy_params,
            )
            return policy_params, target_policy_params, policy_optimizer_state

        # Delayed update
        current_policy_state = (
            training_state.policy_params,
            training_state.target_policy_params,
            training_state.policy_optimizer_state,
        )
        policy_params, target_policy_params, policy_optimizer_state = jax.lax.cond(
            training_state.steps % self._config.policy_delay == 0,
            lambda _: update_policy_step(),
            lambda _: current_policy_state,
            operand=None,
        )

        # Create new training state
        new_training_state = training_state.replace(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            policy_params=policy_params,
            policy_optimizer_state=policy_optimizer_state,
            target_critic_params=target_critic_params,
            target_policy_params=target_policy_params,
            key=key,
            steps=training_state.steps + 1,
        )

        metrics = {
            "actor_loss": policy_loss,
            "critic_loss": critic_loss,
        }

        return new_training_state, replay_buffer, metrics
