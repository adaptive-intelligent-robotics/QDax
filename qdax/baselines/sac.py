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

from qdax.core.neuroevolution.buffers.buffer import (
    QDTransition,
    ReplayBuffer,
    Transition,
)
from qdax.core.neuroevolution.losses.sac_loss import (
    sac_alpha_loss_fn,
    sac_critic_loss_fn,
    sac_policy_loss_fn,
)
from qdax.core.neuroevolution.mdp_utils import TrainingState, get_first_episode
from qdax.core.neuroevolution.networks.sac_networks import make_sac_networks
from qdax.core.neuroevolution.normalization_utils import (
    RunningMeanStdState,
    normalize_with_rmstd,
    update_running_mean_std,
)
from qdax.core.neuroevolution.sac_td3_utils import generate_unroll
from qdax.types import (
    Action,
    Descriptor,
    Mask,
    Metrics,
    Observation,
    Params,
    Reward,
    RNGKey,
)


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
    tau: float = 0.005
    normalize_observations: bool = False
    learning_rate: float = 3e-4
    alpha_init: float = 1.0
    discount: float = 0.97
    reward_scaling: float = 1.0
    critic_hidden_layer_size: tuple = (256, 256)
    policy_hidden_layer_size: tuple = (256, 256)
    fix_alpha: bool = False


class SAC:
    def __init__(self, config: SacConfig, action_size: int) -> None:
        self._config = config
        self._action_size = action_size

        # define the networks
        self._policy, self._critic = make_sac_networks(
            action_size=action_size,
            critic_hidden_layer_size=self._config.critic_hidden_layer_size,
            policy_hidden_layer_size=self._config.policy_hidden_layer_size,
        )

        # define the action distribution
        self._parametric_action_distribution = NormalTanhDistribution(
            event_size=action_size
        )
        self._sample_action_fn = self._parametric_action_distribution.sample

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

        # define initial optimizer states
        optimizer = optax.adam(learning_rate=1.0)
        policy_optimizer_state = optimizer.init(policy_params)
        critic_optimizer_state = optimizer.init(critic_params)

        log_alpha = jnp.asarray(jnp.log(self._config.alpha_init), dtype=jnp.float32)
        alpha_optimizer_state = optimizer.init(log_alpha)

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
            deterministic: whether to select action in a deterministic way.
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
            deterministic: the whether to select action in a deterministic way.
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

    @partial(jax.jit, static_argnames=("self", "env", "deterministic", "evaluation"))
    def play_qd_step_fn(
        self,
        env_state: EnvState,
        training_state: SacTrainingState,
        env: Env,
        deterministic: bool = False,
        evaluation: bool = False,
    ) -> Tuple[EnvState, SacTrainingState, QDTransition]:
        """Plays a step in the environment. Selects an action according to SAC rule and
        performs the environment step.

        Args:
            env_state: the current environment state
            training_state: the SAC training state
            env: the environment
            deterministic: the whether to select action in a deterministic way.
                Defaults to False.
            evaluation: if True, collected transitions are not used to update training
                state. Defaults to False.

        Returns:
            the new environment state
            the new SAC training state
            the played transition
        """

        next_env_state, training_state, transition = self.play_step_fn(
            env_state, training_state, env, deterministic, evaluation
        )
        actions = transition.actions
        next_env_state = env.step(env_state, actions)
        next_obs = next_env_state.obs

        truncations = next_env_state.info["truncation"]

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_obs,
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

        transitions = get_first_episode(transitions)
        true_returns = jnp.nansum(transitions.rewards, axis=0)
        true_return = jnp.mean(true_returns, axis=-1)

        return true_return, true_returns

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "play_step_fn",
            "bd_extraction_fn",
        ),
    )
    def eval_qd_policy_fn(
        self,
        training_state: SacTrainingState,
        eval_env_first_state: EnvState,
        play_step_fn: Callable[
            [EnvState, Params, RNGKey],
            Tuple[EnvState, SacTrainingState, QDTransition],
        ],
        bd_extraction_fn: Callable[[QDTransition, Mask], Descriptor],
    ) -> Tuple[Reward, Descriptor, Reward, Descriptor]:
        """
        Evaluates the agent's policy over an entire episode, across all batched
        environments for QD environments. Averaged BDs are returned as well.


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

        transitions = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), transitions
        )
        masks = jnp.isnan(transitions.rewards)
        bds = bd_extraction_fn(transitions, masks)

        mean_bd = jnp.mean(bds, axis=0)
        return true_return, mean_bd, true_returns, bds

    @partial(jax.jit, static_argnames=("self",))
    def _update_alpha(
        self,
        alpha_lr: float,
        training_state: SacTrainingState,
        transitions: Transition,
        random_key: RNGKey,
    ) -> Tuple[Params, optax.OptState, jnp.ndarray, RNGKey]:
        """Updates the alpha parameter if necessary. Else, it keeps the
        current value.

        Args:
            alpha_lr: alpha learning rate
            training_state: the current training state.
            transitions: a sample of transitions from the replay buffer.
            random_key: a random key to handle stochastic operations.

        Returns:
            New alpha params, optimizer state, loss and a new random key.
        """
        if not self._config.fix_alpha:
            # update alpha
            random_key, subkey = jax.random.split(random_key)
            alpha_loss, alpha_gradient = jax.value_and_grad(sac_alpha_loss_fn)(
                training_state.alpha_params,
                policy_fn=self._policy.apply,
                parametric_action_distribution=self._parametric_action_distribution,
                action_size=self._action_size,
                policy_params=training_state.policy_params,
                transitions=transitions,
                random_key=subkey,
            )
            alpha_optimizer = optax.adam(learning_rate=alpha_lr)
            (alpha_updates, alpha_optimizer_state,) = alpha_optimizer.update(
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

    @partial(jax.jit, static_argnames=("self",))
    def _update_critic(
        self,
        critic_lr: float,
        reward_scaling: float,
        discount: float,
        training_state: SacTrainingState,
        transitions: Transition,
        random_key: RNGKey,
    ) -> Tuple[Params, Params, optax.OptState, jnp.ndarray, RNGKey]:
        """Updates the critic following the method described in the
        Soft Actor Critic paper.

        Args:
            critic_lr: critic learning rate
            reward_scaling: coefficient to scale rewards
            discount: discount factor
            training_state: the current training state.
            transitions: a batch of transitions sampled from the replay buffer.
            random_key: a random key to handle stochastic operations.

        Returns:
            New parameters of the critic and its target. New optimizer state,
            loss and a new random key.
        """
        # update critic
        random_key, subkey = jax.random.split(random_key)
        critic_loss, critic_gradient = jax.value_and_grad(sac_critic_loss_fn)(
            training_state.critic_params,
            policy_fn=self._policy.apply,
            critic_fn=self._critic.apply,
            parametric_action_distribution=self._parametric_action_distribution,
            reward_scaling=reward_scaling,
            discount=discount,
            policy_params=training_state.policy_params,
            target_critic_params=training_state.target_critic_params,
            alpha=jnp.exp(training_state.alpha_params),
            transitions=transitions,
            random_key=subkey,
        )
        critic_optimizer = optax.adam(learning_rate=critic_lr)
        (critic_updates, critic_optimizer_state,) = critic_optimizer.update(
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

    @partial(jax.jit, static_argnames=("self",))
    def _update_actor(
        self,
        policy_lr: float,
        training_state: SacTrainingState,
        transitions: Transition,
        random_key: RNGKey,
    ) -> Tuple[Params, optax.OptState, jnp.ndarray, RNGKey]:
        """Updates the actor parameters following the stochastic
        policy gradient theorem with the method introduced in SAC.

        Args:
            policy_lr: policy learning rate
            training_state: the current training state.
            transitions: a batch of transitions sampled from the replay
                buffer.
            random_key: a random key to handle stochastic operations.

        Returns:
            New params and optimizer state. Current loss. New random key.
        """
        random_key, subkey = jax.random.split(random_key)
        policy_loss, policy_gradient = jax.value_and_grad(sac_policy_loss_fn)(
            training_state.policy_params,
            policy_fn=self._policy.apply,
            critic_fn=self._critic.apply,
            parametric_action_distribution=self._parametric_action_distribution,
            critic_params=training_state.critic_params,
            alpha=jnp.exp(training_state.alpha_params),
            transitions=transitions,
            random_key=subkey,
        )
        policy_optimizer = optax.adam(learning_rate=policy_lr)
        (policy_updates, policy_optimizer_state,) = policy_optimizer.update(
            policy_gradient, training_state.policy_optimizer_state
        )
        policy_params = optax.apply_updates(
            training_state.policy_params, policy_updates
        )

        return policy_params, policy_optimizer_state, policy_loss, random_key

    @partial(jax.jit, static_argnames=("self",))
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

        # update alpha
        (
            alpha_params,
            alpha_optimizer_state,
            alpha_loss,
            random_key,
        ) = self._update_alpha(
            alpha_lr=self._config.learning_rate,
            training_state=training_state,
            transitions=transitions,
            random_key=random_key,
        )

        # update critic
        (
            critic_params,
            target_critic_params,
            critic_optimizer_state,
            critic_loss,
            random_key,
        ) = self._update_critic(
            critic_lr=self._config.learning_rate,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            training_state=training_state,
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
            policy_lr=self._config.learning_rate,
            training_state=training_state,
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
