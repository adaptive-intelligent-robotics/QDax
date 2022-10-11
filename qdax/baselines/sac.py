"""
A collection of functions and classes that define the algorithm Soft Actor Critic
(SAC), ref: https://arxiv.org/abs/1801.01290
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from brax.envs import Env
from brax.envs import State as EnvState
from brax.training.distribution import NormalTanhDistribution

from qdax.core.neuroevolution.buffers.buffer import ReplayBuffer, Transition
from qdax.core.neuroevolution.losses.sac_loss import make_sac_loss_fn
from qdax.core.neuroevolution.mdp_utils import TrainingState, get_first_episode
from qdax.core.neuroevolution.networks.sac_networks import make_sac_networks
from qdax.core.neuroevolution.normalization_utils import (
    RunningMeanStdState,
    normalize_with_rmstd,
    update_running_mean_std,
)
from qdax.core.neuroevolution.sac_utils import generate_unroll
from qdax.environments import CompletedEvalWrapper
from qdax.types import Action, Metrics, Observation, Params, Reward, RNGKey


class SacTrainingState(TrainingState):
    """Training state for the SAC algorithm"""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    critic_optimizer_state: optax.OptState
    critic_params: Params
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    target_critic_params: Params
    random_key: RNGKey
    steps: jnp.ndarray
    normalization_running_stats: RunningMeanStdState


@dataclass
class SacConfig:
    """Configuration for the SAC algorithm."""

    batch_size: int
    episode_length: int
    grad_updates_per_step: float
    tau: float = 0.005
    normalize_observations: bool = False
    learning_rate: float = 3e-4
    alpha_init: float = 1.0
    discount: float = 0.97
    reward_scaling: float = 1.0
    hidden_layer_sizes: tuple = (256, 256)
    fix_alpha: bool = False


class SAC:
    def __init__(self, config: SacConfig, action_size: int) -> None:
        self._config = config

        # define the networks
        self._policy, self._critic = make_sac_networks(
            action_size=action_size, hidden_layer_sizes=self._config.hidden_layer_sizes
        )

        # define the action distribution
        parametric_action_distribution = NormalTanhDistribution(event_size=action_size)
        self._sample_action_fn = parametric_action_distribution.sample

        # define the losses
        (
            self._alpha_loss_fn,
            self._policy_loss_fn,
            self._critic_loss_fn,
        ) = make_sac_loss_fn(
            policy_fn=self._policy.apply,
            critic_fn=self._critic.apply,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            action_size=action_size,
            parametric_action_distribution=parametric_action_distribution,
        )

        # define the optimizers
        self._policy_optimizer = optax.adam(learning_rate=self._config.learning_rate)
        self._critic_optimizer = optax.adam(learning_rate=self._config.learning_rate)
        self._alpha_optimizer = optax.adam(learning_rate=self._config.learning_rate)

    def init(
        self, random_key: RNGKey, action_size: int, observation_size: int
    ) -> SacTrainingState:
        """Initialise the training state of the algorithm.

        Args:
            random_key: a jax random key
            action_size: the size of the environment's action space
            observation_size: the size of the environment's observation space

        Returns:
            the initial training state of SAC
        """

        # define policy and critic params
        dummy_obs = jnp.zeros((1, observation_size))
        dummy_action = jnp.zeros((1, action_size))

        random_key, subkey = jax.random.split(random_key)
        policy_params = self._policy.init(subkey, dummy_obs)

        random_key, subkey = jax.random.split(random_key)
        critic_params = self._critic.init(subkey, dummy_obs, dummy_action)

        target_critic_params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x.copy()), critic_params
        )

        # define intitial optimizer states
        policy_optimizer_state = self._policy_optimizer.init(policy_params)
        critic_optimizer_state = self._critic_optimizer.init(critic_params)

        log_alpha = jnp.asarray(jnp.log(self._config.alpha_init), dtype=jnp.float32)
        alpha_optimizer_state = self._alpha_optimizer.init(log_alpha)

        # create and retrieve the training state
        training_state = SacTrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=log_alpha,
            target_critic_params=target_critic_params,
            normalization_running_stats=RunningMeanStdState(
                mean=jnp.zeros(
                    observation_size,
                ),
                var=jnp.ones(
                    observation_size,
                ),
                count=jnp.zeros(()),
            ),
            random_key=random_key,
            steps=jnp.array(0),
        )

        return training_state

    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def select_action(
        self,
        obs: Observation,
        policy_params: Params,
        random_key: RNGKey,
        deterministic: bool = False,
    ) -> Tuple[Action, RNGKey]:
        """Selects an action acording to SAC policy.

        Args:
            obs: agent observation(s)
            policy_params: parameters of the agent's policy
            random_key: jax random key
            deterministic: whether or not to select action in a deterministic way.
                Defaults to False.

        Returns:
            The selected action and a new random key.
        """

        dist_params = self._policy.apply(policy_params, obs)
        if not deterministic:
            random_key, key_sample = jax.random.split(random_key)
            actions = self._sample_action_fn(dist_params, key_sample)

        else:
            # The first half of parameters is for mean and the second half for variance
            actions = jax.nn.tanh(dist_params[..., : dist_params.shape[-1] // 2])

        return actions, random_key

    @partial(jax.jit, static_argnames=("self", "env", "deterministic", "evaluation"))
    def play_step_fn(
        self,
        env_state: EnvState,
        training_state: SacTrainingState,
        env: Env,
        deterministic: bool = False,
        evaluation: bool = False,
    ) -> Tuple[EnvState, SacTrainingState, Transition]:
        """Plays a step in the environment. Selects an action according to SAC rule and
        performs the environment step.

        Args:
            env_state: the current environment state
            training_state: the SAC training state
            env: the environment
            deterministic: the whether or not to select action in a deterministic way.
                Defaults to False.
            evaluation: if True, collected transitions are not used to update training
                state. Defaults to False.

        Returns:
            the new environment state
            the new SAC training state
            the played transition
        """
        random_key = training_state.random_key
        policy_params = training_state.policy_params
        obs = env_state.obs

        if self._config.normalize_observations:
            normalized_obs = normalize_with_rmstd(
                obs, training_state.normalization_running_stats
            )
            normalization_running_stats = update_running_mean_std(
                training_state.normalization_running_stats, obs
            )

        else:
            normalized_obs = obs
            normalization_running_stats = training_state.normalization_running_stats

        actions, random_key = self.select_action(
            obs=normalized_obs,
            policy_params=policy_params,
            random_key=random_key,
            deterministic=deterministic,
        )

        if not evaluation:
            training_state = training_state.replace(
                random_key=random_key,
                normalization_running_stats=normalization_running_stats,
            )
        else:
            training_state = training_state.replace(
                random_key=random_key,
            )

        next_env_state = env.step(env_state, actions)
        next_obs = next_env_state.obs

        truncations = next_env_state.info["truncation"]
        transition = Transition(
            obs=env_state.obs,
            next_obs=next_obs,
            rewards=next_env_state.reward,
            dones=next_env_state.done,
            actions=actions,
            truncations=truncations,
        )

        return (
            next_env_state,
            training_state,
            transition,
        )

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "play_step_fn",
        ),
    )
    def eval_policy_fn(
        self,
        training_state: SacTrainingState,
        eval_env_first_state: EnvState,
        play_step_fn: Callable[
            [EnvState, Params, RNGKey],
            Tuple[EnvState, SacTrainingState, Transition],
        ],
    ) -> Tuple[Reward, Reward]:
        """Evaluates the agent's policy over an entire episode, across all batched
        environments.


        Args:
            training_state: the SAC training state
            eval_env_first_state: the initial state for evaluation
            play_step_fn: the play_step function used to collect the evaluation episode

        Returns:
            the true return averaged over batch dimension, shape: (1,)
            the true return per environment, shape: (env_batch_size,)

        """

        state, training_state, transitions = generate_unroll(
            init_state=eval_env_first_state,
            training_state=training_state,
            episode_length=self._config.episode_length,
            play_step_fn=play_step_fn,
        )

        eval_metrics_key = CompletedEvalWrapper.STATE_INFO_KEY
        true_return = (
            state.info[eval_metrics_key].completed_episodes_metrics["reward"]
            / state.info[eval_metrics_key].completed_episodes
        )

        transitions = get_first_episode(transitions)

        true_returns = jnp.nansum(transitions.rewards, axis=0)

        return true_return, true_returns

    @partial(jax.jit, static_argnames=("self"))
    def _update_alpha(
        self,
        training_state: SacTrainingState,
        transitions: Transition,
        random_key: RNGKey,
    ) -> Tuple[Params, optax.OptState, jnp.ndarray, RNGKey]:
        """Updates the alpha parameter if necessary. Else, it keeps the
        current value.

        Args:
            training_state: the current training state.
            transitions: a sample of transitions from the replay buffer.
            random_key: a random key to handle stochastic operations.

        Returns:
            New alpha params, optimizer state, loss and a new random key.
        """
        if not self._config.fix_alpha:
            # update alpha
            random_key, subkey = jax.random.split(random_key)
            alpha_loss, alpha_gradient = jax.value_and_grad(self._alpha_loss_fn)(
                training_state.alpha_params,
                training_state.policy_params,
                transitions=transitions,
                random_key=subkey,
            )

            (alpha_updates, alpha_optimizer_state,) = self._alpha_optimizer.update(
                alpha_gradient, training_state.alpha_optimizer_state
            )
            alpha_params = optax.apply_updates(
                training_state.alpha_params, alpha_updates
            )
        else:
            alpha_params = training_state.alpha_params
            alpha_optimizer_state = training_state.alpha_optimizer_state
            alpha_loss = jnp.array(0.0)

        return alpha_params, alpha_optimizer_state, alpha_loss, random_key

    @partial(jax.jit, static_argnames=("self"))
    def _update_critic(
        self,
        training_state: SacTrainingState,
        alpha: Params,
        transitions: Transition,
        random_key: RNGKey,
    ) -> Tuple[Params, Params, optax.OptState, jnp.ndarray, RNGKey]:
        """Updates the critic following the method described in the
        Soft Actor Critic paper.

        Args:
            training_state: the current training state.
            alpha: the alpha parameter that controls the importance of
                the entropy term.
            transitions: a batch of transitions sampled from the replay buffer.
            random_key: a random key to handle stochastic operations.

        Returns:
            New parameters of the critic and its target. New optimizer state,
            loss and a new random key.
        """
        # update critic
        random_key, subkey = jax.random.split(random_key)
        critic_loss, critic_gradient = jax.value_and_grad(self._critic_loss_fn)(
            training_state.critic_params,
            training_state.policy_params,
            training_state.target_critic_params,
            alpha,
            transitions=transitions,
            random_key=subkey,
        )

        (critic_updates, critic_optimizer_state,) = self._critic_optimizer.update(
            critic_gradient, training_state.critic_optimizer_state
        )
        critic_params = optax.apply_updates(
            training_state.critic_params, critic_updates
        )
        target_critic_params = jax.tree_util.tree_map(
            lambda x1, x2: (1.0 - self._config.tau) * x1 + self._config.tau * x2,
            training_state.target_critic_params,
            critic_params,
        )

        return (
            critic_params,
            target_critic_params,
            critic_optimizer_state,
            critic_loss,
            random_key,
        )

    @partial(jax.jit, static_argnames=("self"))
    def _update_actor(
        self,
        training_state: SacTrainingState,
        alpha: Params,
        transitions: Transition,
        random_key: RNGKey,
    ) -> Tuple[Params, optax.OptState, jnp.ndarray, RNGKey]:
        """Updates the actor parameters following the stochastic
        policy gradient theorem with the method introduced in SAC.

        Args:
            training_state: the currrent training state.
            alpha: the alpha parameter that controls the importance
                of the entropy term.
            transitions: a batch of transitions sampled from the replay
                buffer.
            random_key: a random key to handle stochastic operations.

        Returns:
            New params and optimizer state. Current loss. New random key.
        """
        random_key, subkey = jax.random.split(random_key)
        policy_loss, policy_gradient = jax.value_and_grad(self._policy_loss_fn)(
            training_state.policy_params,
            training_state.critic_params,
            alpha,
            transitions=transitions,
            random_key=subkey,
        )
        (policy_updates, policy_optimizer_state,) = self._policy_optimizer.update(
            policy_gradient, training_state.policy_optimizer_state
        )
        policy_params = optax.apply_updates(
            training_state.policy_params, policy_updates
        )

        return policy_params, policy_optimizer_state, policy_loss, random_key

    @partial(jax.jit, static_argnames=("self"))
    def update(
        self,
        training_state: SacTrainingState,
        replay_buffer: ReplayBuffer,
    ) -> Tuple[SacTrainingState, ReplayBuffer, Metrics]:
        """Performs a training step to update the policy and the critic parameters.

        Args:
            training_state: the current SAC training state
            replay_buffer: the replay buffer

        Returns:
            the updated SAC training state
            the replay buffer
            the training metrics
        """

        # sample a batch of transitions in the buffer
        random_key = training_state.random_key
        transitions, random_key = replay_buffer.sample(
            random_key,
            sample_size=self._config.batch_size,
        )

        # normalise observations if necessary
        if self._config.normalize_observations:
            normalization_running_stats = training_state.normalization_running_stats
            normalized_obs = normalize_with_rmstd(
                transitions.obs, normalization_running_stats
            )
            normalized_next_obs = normalize_with_rmstd(
                transitions.next_obs, normalization_running_stats
            )
            transitions = transitions.replace(
                obs=normalized_obs, next_obs=normalized_next_obs
            )

        # udpate alpha
        (
            alpha_params,
            alpha_optimizer_state,
            alpha_loss,
            random_key,
        ) = self._update_alpha(
            training_state=training_state,
            transitions=transitions,
            random_key=random_key,
        )

        # use the previous alpha
        alpha = jnp.exp(training_state.alpha_params)

        # update critic
        (
            critic_params,
            target_critic_params,
            critic_optimizer_state,
            critic_loss,
            random_key,
        ) = self._update_critic(
            training_state=training_state,
            alpha=alpha,
            transitions=transitions,
            random_key=random_key,
        )

        # update actor
        (
            policy_params,
            policy_optimizer_state,
            policy_loss,
            random_key,
        ) = self._update_actor(
            training_state=training_state,
            alpha=alpha,
            transitions=transitions,
            random_key=random_key,
        )

        # create new training state
        new_training_state = SacTrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            normalization_running_stats=training_state.normalization_running_stats,
            target_critic_params=target_critic_params,
            random_key=random_key,
            steps=training_state.steps + 1,
        )
        metrics = {
            "actor_loss": policy_loss,
            "critic_loss": critic_loss,
            "alpha_loss": alpha_loss,
            "obs_mean": jnp.mean(transitions.obs),
            "obs_std": jnp.std(transitions.obs),
        }
        return new_training_state, replay_buffer, metrics
