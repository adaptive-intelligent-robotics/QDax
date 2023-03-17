"""
Implementation of the Population Based Training (PBT) algorithm
(https://arxiv.org/abs/1711.09846) to tune the hyperparameters of the SAC algorithm.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from brax.envs import Env
from brax.envs import State as EnvState

from qdax.baselines.pbt import PBTTrainingState
from qdax.baselines.sac import SAC, SacConfig, SacTrainingState
from qdax.core.neuroevolution.buffers.buffer import (
    QDTransition,
    ReplayBuffer,
    Transition,
)
from qdax.core.neuroevolution.normalization_utils import normalize_with_rmstd
from qdax.core.neuroevolution.sac_td3_utils import do_iteration_fn
from qdax.types import Descriptor, Mask, Metrics, RNGKey


class PBTSacTrainingState(PBTTrainingState, SacTrainingState):
    """Training state for the PBT-SAC algorithm"""

    # Add hyperparameters as part of the state for PBT
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
    """Configuration for the PBT-SAC algorithm."""

    batch_size: int
    episode_length: int
    tau: float = 0.005
    normalize_observations: bool = False
    alpha_init: float = 1.0
    policy_hidden_layer_size: tuple = (256, 256)
    critic_hidden_layer_size: tuple = (256, 256)
    fix_alpha: bool = False


class PBTSAC(SAC):
    def __init__(self, config: PBTSacConfig, action_size: int) -> None:

        sac_config = SacConfig(
            batch_size=config.batch_size,
            episode_length=config.episode_length,
            tau=config.tau,
            normalize_observations=config.normalize_observations,
            alpha_init=config.alpha_init,
            policy_hidden_layer_size=config.policy_hidden_layer_size,
            critic_hidden_layer_size=config.critic_hidden_layer_size,
            fix_alpha=config.fix_alpha,
            # unused default values for parameters that will be learnt as part of PBT
            learning_rate=3e-4,
            discount=0.97,
            reward_scaling=1.0,
        )
        SAC.__init__(self, config=sac_config, action_size=action_size)

    def init(
        self, random_key: RNGKey, action_size: int, observation_size: int
    ) -> PBTSacTrainingState:
        """Initialise the training state of the algorithm.

        Args:
            random_key: a jax random key
            action_size: the size of the environment's action space
            observation_size: the size of the environment's observation space

        Returns:
            the initial training state of PBT-SAC
        """

        sac_training_state = SAC.init(self, random_key, action_size, observation_size)

        training_state = PBTSacTrainingState(
            policy_optimizer_state=sac_training_state.policy_optimizer_state,
            policy_params=sac_training_state.policy_params,
            critic_optimizer_state=sac_training_state.critic_optimizer_state,
            critic_params=sac_training_state.critic_params,
            alpha_optimizer_state=sac_training_state.alpha_optimizer_state,
            alpha_params=sac_training_state.alpha_params,
            target_critic_params=sac_training_state.target_critic_params,
            normalization_running_stats=sac_training_state.normalization_running_stats,
            random_key=sac_training_state.random_key,
            steps=sac_training_state.steps,
            discount=None,
            policy_lr=None,
            critic_lr=None,
            alpha_lr=None,
            reward_scaling=None,
        )

        # Sample hyper-params
        training_state = PBTSacTrainingState.resample_hyperparams(training_state)

        return training_state  # type: ignore

    @partial(jax.jit, static_argnames=("self"))
    def update(
        self,
        training_state: PBTSacTrainingState,
        replay_buffer: ReplayBuffer,
    ) -> Tuple[PBTSacTrainingState, ReplayBuffer, Metrics]:
        """Performs a training step to update the policy and the critic parameters.

        Args:
            training_state: the current PBT-SAC training state
            replay_buffer: the replay buffer

        Returns:
            the updated PBT-SAC training state
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
            alpha_lr=training_state.alpha_lr,
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
            critic_lr=training_state.critic_lr,
            reward_scaling=training_state.reward_scaling,
            discount=training_state.discount,
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
            policy_lr=training_state.policy_lr,
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
        Returns a function to initialize the population.

        Args:
            population_size: size of the population.
            action_size: action space size.
            observation_size: observation space size.
            buffer_size: replay buffer size.

        Returns:
            a function that takes as input a random key and returns a new random
            key, the PBT population training state and the replay buffers
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
        Returns the function the evaluation the PBT population.

        Args:
            eval_env: evaluation environment. Might be different from training env
                if needed.

        Returns:
            The function to evaluate the population. It takes as input the population
            training state as well as first eval environment states and returns the
            population agents mean returns over episodes as well as all returns from all
            agents over all episodes.
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
        Returns the function the evaluation the PBT population.

        Args:
            eval_env: evaluation environment. Might be different from training env
                if needed.
            bd_extraction_fn: function to extract the bd from an episode.

        Returns:
            The function to evaluate the population. It takes as input the population
            training state as well as first eval environment states and returns the
            population agents mean returns and mean bds over episodes as well as all
            returns and bds from all agents over all episodes.
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
        grad_updates_per_step: float,
    ) -> Callable:
        """
        Returns the function to update the population of agents.

        Args:
            env: training environment.
            num_iterations: number of training iterations to perform.
            env_batch_size: number of batched environments.
            grad_updates_per_step: number of gradient to apply per step in the
                environment.

        Returns:
            the function to update the population which takes as input the population
            training state, environment starting states and replay buffers and returns
            updated training states, environment states, replay buffers and metrics.
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
