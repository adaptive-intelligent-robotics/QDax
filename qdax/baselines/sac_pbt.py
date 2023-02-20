"""
A collection of functions and classes that define the algorithm Soft Actor Critic
(SAC), ref: https://arxiv.org/abs/1801.01290
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from brax.envs import Env
from brax.envs import State as EnvState
from brax.training.distribution import NormalTanhDistribution

from qdax.baselines.pbt import PBTTrainingState
from qdax.baselines.sac import SacTrainingState
from qdax.core.neuroevolution.buffers.buffer import (
    QDTransition,
    ReplayBuffer,
    Transition,
)
from qdax.core.neuroevolution.mdp_utils import get_first_episode
from qdax.core.neuroevolution.networks.sac_networks import make_sac_networks
from qdax.core.neuroevolution.normalization_utils import (
    RunningMeanStdState,
    normalize_with_rmstd,
    update_running_mean_std,
)
from qdax.core.neuroevolution.sac_utils import do_iteration_fn, generate_unroll
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


class PBTSacTrainingState(PBTTrainingState, SacTrainingState):
    """Training state for the SAC algorithm"""

    # Add hyper-parameters as part of the state for PBT
    discount: float
    policy_lr: float
    critic_lr: float
    alpha_lr: float
    reward_scaling: float

    @classmethod
    @partial(jax.jit, static_argnames=("cls",))
    def init_optimizers_states(
        cls,
        training_state: "PBTSacTrainingState",
    ) -> "PBTSacTrainingState":
        optimizer_init = optax.adam(learning_rate=1.0).init
        policy_params = training_state.policy_params
        critic_params = training_state.critic_params
        alpha_params = training_state.alpha_params
        target_critic_params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x.copy()), critic_params
        )
        return training_state.replace(  # type: ignore
            target_critic_params=target_critic_params,
            policy_optimizer_state=optimizer_init(policy_params),
            critic_optimizer_state=optimizer_init(critic_params),
            alpha_optimizer_state=optimizer_init(alpha_params),
        )

    @classmethod
    @partial(jax.jit, static_argnames=("cls",))
    def empty_optimizers_states(
        cls,
        training_state: "PBTSacTrainingState",
    ) -> "PBTSacTrainingState":
        return training_state.replace(  # type: ignore
            target_critic_params=jnp.empty(shape=(1,), dtype=jnp.float32),
            policy_optimizer_state=jnp.empty(shape=(1,), dtype=jnp.float32),
            critic_optimizer_state=jnp.empty(shape=(1,), dtype=jnp.float32),
            alpha_optimizer_state=jnp.empty(shape=(1,), dtype=jnp.float32),
        )

    @classmethod
    @partial(jax.jit, static_argnames=("cls",))
    def resample_hyperparams(
        cls,
        training_state: "PBTSacTrainingState",
    ) -> "PBTSacTrainingState":

        random_key = training_state.random_key
        random_key, sub_key = jax.random.split(random_key)
        discount = jax.random.uniform(sub_key, shape=(), minval=0.9, maxval=1.0)

        random_key, sub_key = jax.random.split(random_key)
        policy_lr = jax.random.uniform(sub_key, shape=(), minval=3e-5, maxval=3e-3)

        random_key, sub_key = jax.random.split(random_key)
        critic_lr = jax.random.uniform(sub_key, shape=(), minval=3e-5, maxval=3e-3)

        random_key, sub_key = jax.random.split(random_key)
        alpha_lr = jax.random.uniform(sub_key, shape=(), minval=3e-5, maxval=3e-3)

        random_key, sub_key = jax.random.split(random_key)
        reward_scaling = jax.random.uniform(sub_key, shape=(), minval=0.1, maxval=10.0)

        return training_state.replace(  # type: ignore
            discount=discount,
            policy_lr=policy_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            reward_scaling=reward_scaling,
            random_key=random_key,
        )


@dataclass
class PBTSacConfig:
    """Configuration for the SAC algorithm."""

    batch_size: int
    episode_length: int
    tau: float = 0.005
    normalize_observations: bool = False
    alpha_init: float = 1.0
    hidden_layer_sizes: tuple = (256, 256)
    fix_alpha: bool = False


class PBTSAC:
    def __init__(self, config: PBTSacConfig, action_size: int) -> None:
        self._config = config

        # define the networks
        self._policy, self._critic = make_sac_networks(
            action_size=action_size, hidden_layer_sizes=self._config.hidden_layer_sizes
        )

        # define the action distribution
        self._parametric_action_distribution = NormalTanhDistribution(
            event_size=action_size
        )
        self._sample_action_fn = self._parametric_action_distribution.sample
        self._action_size = action_size

    def init(
        self, random_key: RNGKey, action_size: int, observation_size: int
    ) -> PBTSacTrainingState:
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
        running_stats = (
            RunningMeanStdState(
                mean=jnp.zeros(
                    observation_size,
                ),
                var=jnp.ones(
                    observation_size,
                ),
                count=jnp.zeros(()),
            ),
        )
        training_state = PBTSacTrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=log_alpha,
            target_critic_params=target_critic_params,
            normalization_running_stats=running_stats,
            random_key=random_key,
            steps=jnp.array(0),
            discount=None,
            policy_lr=None,
            critic_lr=None,
            alpha_lr=None,
            reward_scaling=None,
        )

        # Sample hyper-params
        training_state = PBTSacTrainingState.resample_hyperparams(training_state)

        return training_state  # type: ignore

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
        training_state: PBTSacTrainingState,
        env: Env,
        deterministic: bool = False,
        evaluation: bool = False,
    ) -> Tuple[EnvState, PBTSacTrainingState, Transition]:
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

    @partial(jax.jit, static_argnames=("self", "env", "deterministic", "evaluation"))
    def play_qd_step_fn(
        self,
        env_state: EnvState,
        training_state: PBTSacTrainingState,
        env: Env,
        deterministic: bool = False,
        evaluation: bool = False,
    ) -> Tuple[EnvState, PBTSacTrainingState, QDTransition]:
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
        training_state: PBTSacTrainingState,
        eval_env_first_state: EnvState,
        play_step_fn: Callable[
            [EnvState, Params, RNGKey],
            Tuple[EnvState, PBTSacTrainingState, Transition],
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
        training_state: PBTSacTrainingState,
        eval_env_first_state: EnvState,
        play_step_fn: Callable[
            [EnvState, Params, RNGKey],
            Tuple[EnvState, PBTSacTrainingState, QDTransition],
        ],
        bd_extraction_fn: Callable[[QDTransition, Mask], Descriptor],
    ) -> Tuple[Reward, Descriptor, Reward, Descriptor]:
        """Evaluates the agent's policy over an entire episode, across all batched
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
    def _policy_loss_fn(
        self,
        policy_params: Params,
        critic_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        random_key: RNGKey,
    ) -> jnp.ndarray:

        dist_params = self._policy.apply(policy_params, transitions.obs)
        action = self._parametric_action_distribution.sample_no_postprocessing(
            dist_params, random_key
        )
        log_prob = self._parametric_action_distribution.log_prob(dist_params, action)
        action = self._parametric_action_distribution.postprocess(action)
        q_action = self._critic.apply(critic_params, transitions.obs, action)
        min_q = jnp.min(q_action, axis=-1)
        actor_loss = alpha * log_prob - min_q

        return jnp.mean(actor_loss)

    @partial(jax.jit, static_argnames=("self"))
    def _critic_loss_fn(
        self,
        critic_params: Params,
        policy_params: Params,
        target_critic_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        random_key: RNGKey,
        reward_scaling: float,
        discount: float,
    ) -> jnp.ndarray:

        q_old_action = self._critic.apply(
            critic_params, transitions.obs, transitions.actions
        )
        next_dist_params = self._policy.apply(policy_params, transitions.next_obs)
        next_action = self._parametric_action_distribution.sample_no_postprocessing(
            next_dist_params, random_key
        )
        next_log_prob = self._parametric_action_distribution.log_prob(
            next_dist_params, next_action
        )
        next_action = self._parametric_action_distribution.postprocess(next_action)
        next_q = self._critic.apply(
            target_critic_params, transitions.next_obs, next_action
        )

        next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob

        target_q = jax.lax.stop_gradient(
            transitions.rewards * reward_scaling
            + (1.0 - transitions.dones) * discount * next_v
        )

        q_error = q_old_action - jnp.expand_dims(target_q, -1)
        q_error *= jnp.expand_dims(1 - transitions.truncations, -1)
        q_loss = 0.5 * jnp.mean(jnp.square(q_error))

        return q_loss

    @partial(jax.jit, static_argnames=("self"))
    def _alpha_loss_fn(
        self,
        log_alpha: jnp.ndarray,
        policy_params: Params,
        transitions: Transition,
        random_key: RNGKey,
    ) -> jnp.ndarray:
        """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""

        target_entropy = -0.5 * self._action_size
        dist_params = self._policy.apply(policy_params, transitions.obs)
        action = self._parametric_action_distribution.sample_no_postprocessing(
            dist_params, random_key
        )
        log_prob = self._parametric_action_distribution.log_prob(dist_params, action)
        # TODO: check line below that seems to be unused
        action = self._parametric_action_distribution.postprocess(action)
        alpha = jnp.exp(log_alpha)
        alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)

        loss = jnp.mean(alpha_loss)
        return loss

    @partial(jax.jit, static_argnames=("self"))
    def _update_alpha(
        self,
        training_state: PBTSacTrainingState,
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
            alpha_optimizer = optax.adam(learning_rate=training_state.alpha_lr)
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

    @partial(jax.jit, static_argnames=("self"))
    def _update_critic(
        self,
        training_state: PBTSacTrainingState,
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
        alpha = jnp.exp(training_state.alpha_params)
        critic_loss, critic_gradient = jax.value_and_grad(self._critic_loss_fn)(
            training_state.critic_params,
            training_state.policy_params,
            training_state.target_critic_params,
            alpha,
            transitions=transitions,
            random_key=subkey,
            discount=training_state.discount,
            reward_scaling=training_state.reward_scaling,
        )
        critic_optimizer = optax.adam(learning_rate=training_state.critic_lr)
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

    @partial(jax.jit, static_argnames=("self"))
    def _update_actor(
        self,
        training_state: PBTSacTrainingState,
        transitions: Transition,
        random_key: RNGKey,
    ) -> Tuple[Params, optax.OptState, jnp.ndarray, RNGKey]:
        """Updates the actor parameters following the stochastic
        policy gradient theorem with the method introduced in SAC.

        Args:
            training_state: the currrent training state.
            transitions: a batch of transitions sampled from the replay
                buffer.
            random_key: a random key to handle stochastic operations.

        Returns:
            New params and optimizer state. Current loss. New random key.
        """
        random_key, subkey = jax.random.split(random_key)
        alpha = jnp.exp(training_state.alpha_params)
        policy_loss, policy_gradient = jax.value_and_grad(self._policy_loss_fn)(
            training_state.policy_params,
            training_state.critic_params,
            alpha,
            transitions=transitions,
            random_key=subkey,
        )
        policy_optimizer = optax.adam(learning_rate=training_state.policy_lr)
        (policy_updates, policy_optimizer_state,) = policy_optimizer.update(
            policy_gradient, training_state.policy_optimizer_state
        )
        policy_params = optax.apply_updates(
            training_state.policy_params, policy_updates
        )

        return policy_params, policy_optimizer_state, policy_loss, random_key

    @partial(jax.jit, static_argnames=("self"))
    def update(
        self,
        training_state: PBTSacTrainingState,
        replay_buffer: ReplayBuffer,
    ) -> Tuple[PBTSacTrainingState, ReplayBuffer, Metrics]:
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
            training_state=training_state,
            transitions=transitions,
            random_key=random_key,
        )

        # create new training state
        new_training_state = PBTSacTrainingState(
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
            discount=training_state.discount,
            policy_lr=training_state.policy_lr,
            critic_lr=training_state.critic_lr,
            alpha_lr=training_state.alpha_lr,
            reward_scaling=training_state.reward_scaling,
        )
        metrics = {
            "actor_loss": policy_loss,
            "critic_loss": critic_loss,
            "alpha_loss": alpha_loss,
            "obs_mean": jnp.mean(transitions.obs),
            "obs_std": jnp.std(transitions.obs),
        }
        return new_training_state, replay_buffer, metrics

    def get_init_fn(
        self,
        population_size: int,
        action_size: int,
        observation_size: int,
        buffer_size: int,
    ) -> Callable:
        """
        TODO: docstring
        Args:
            population_size:
            action_size:
            observation_size:
            buffer_size:

        Returns:

        """

        def _init_fn(
            random_key: RNGKey,
        ) -> Tuple[RNGKey, PBTSacTrainingState, ReplayBuffer]:

            random_key, *keys = jax.random.split(random_key, num=1 + population_size)
            keys = jnp.stack(keys)

            init_dummy_transition = partial(
                Transition.init_dummy,
                observation_dim=observation_size,
                action_dim=action_size,
            )
            init_dummy_transition = jax.vmap(
                init_dummy_transition, axis_size=population_size
            )
            dummy_transitions = init_dummy_transition()

            replay_buffer_init = partial(
                ReplayBuffer.init,
                buffer_size=buffer_size,
            )
            replay_buffer_init = jax.vmap(replay_buffer_init)
            replay_buffers = replay_buffer_init(transition=dummy_transitions)
            agent_init = partial(
                self.init, action_size=action_size, observation_size=observation_size
            )
            training_states = jax.vmap(agent_init)(keys)
            return random_key, training_states, replay_buffers

        return _init_fn

    def get_eval_fn(
        self,
        eval_env: Env,
    ) -> Callable:
        """
        TODO: docstring
        Args:
            eval_env:

        Returns:

        """
        play_eval_step = partial(
            self.play_step_fn,
            env=eval_env,
            deterministic=True,
        )

        eval_policy = partial(
            self.eval_policy_fn,
            play_step_fn=play_eval_step,
        )
        return jax.vmap(eval_policy)  # type: ignore

    def get_eval_qd_fn(
        self,
        eval_env: Env,
        bd_extraction_fn: Callable[[QDTransition, Mask], Descriptor],
    ) -> Callable:
        """
        TODO: docstring
        Args:
            eval_env:
            bd_extraction_fn:

        Returns:

        """
        play_eval_step = partial(
            self.play_qd_step_fn,
            env=eval_env,
            deterministic=True,
        )

        eval_policy = partial(
            self.eval_qd_policy_fn,
            play_step_fn=play_eval_step,
            bd_extraction_fn=bd_extraction_fn,
        )
        return jax.vmap(eval_policy)  # type: ignore

    def get_train_fn(
        self,
        env: Env,
        num_iterations: int,
        env_batch_size: int,
        grad_updates_per_step: int,
    ) -> Callable:
        """
        TODO: docstring
        Args:
            env:
            num_training_steps:

        Returns:

        """
        play_step = partial(
            self.play_step_fn,
            env=env,
            deterministic=False,
        )

        do_iteration = partial(
            do_iteration_fn,
            env_batch_size=env_batch_size,
            grad_updates_per_step=grad_updates_per_step,
            play_step_fn=play_step,
            update_fn=self.update,
        )

        def _scan_do_iteration(
            carry: Tuple[PBTSacTrainingState, EnvState, ReplayBuffer],
            unused_arg: Any,
        ) -> Tuple[Tuple[PBTSacTrainingState, EnvState, ReplayBuffer], Any]:
            (
                training_state,
                env_state,
                replay_buffer,
                metrics,
            ) = do_iteration(*carry)
            return (training_state, env_state, replay_buffer), metrics

        def train_fn(
            training_state: PBTSacTrainingState,
            env_state: EnvState,
            replay_buffer: ReplayBuffer,
        ) -> Tuple[Tuple[PBTSacTrainingState, EnvState, ReplayBuffer], Metrics]:
            (training_state, env_state, replay_buffer), metrics = jax.lax.scan(
                _scan_do_iteration,
                (training_state, env_state, replay_buffer),
                None,
                length=num_iterations,
            )
            return (training_state, env_state, replay_buffer), metrics

        return jax.vmap(train_fn)  # type: ignore
