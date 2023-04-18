"""
Implementation of the Population Based Training (PBT) algorithm
(https://arxiv.org/abs/1711.09846) to tune the hyperparameters of the TD3 algorithm.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Tuple

import jax
import optax
from brax.envs import Env
from brax.envs import State as EnvState
from jax import numpy as jnp

from qdax.baselines.pbt import PBTTrainingState
from qdax.baselines.td3 import TD3, TD3Config, TD3TrainingState
from qdax.core.neuroevolution.buffers.buffer import (
    QDTransition,
    ReplayBuffer,
    Transition,
)
from qdax.core.neuroevolution.losses.td3_loss import (
    td3_critic_loss_fn,
    td3_policy_loss_fn,
)
from qdax.core.neuroevolution.sac_td3_utils import do_iteration_fn
from qdax.types import Descriptor, Mask, Metrics, Params, RNGKey


class PBTTD3TrainingState(PBTTrainingState, TD3TrainingState):
    """Contains training state for the learner."""

    # Add hyperparameters as part of the state for PBT
    discount: float
    critic_lr: float
    policy_lr: float
    noise_clip: float
    policy_noise: float
    expl_noise: float

    @classmethod
    @partial(jax.jit, static_argnames=("cls",))
    def init_optimizers_states(
        cls,
        training_state: "PBTTD3TrainingState",
    ) -> "PBTTD3TrainingState":
        optimizer_init = optax.adam(learning_rate=1.0).init
        policy_params = training_state.policy_params
        critic_params = training_state.critic_params
        target_critic_params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x.copy()), critic_params
        )
        target_policy_params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x.copy()), policy_params
        )
        return training_state.replace(  # type: ignore
            target_critic_params=target_critic_params,
            target_policy_params=target_policy_params,
            policy_optimizer_state=optimizer_init(policy_params),
            critic_optimizer_state=optimizer_init(critic_params),
        )

    @classmethod
    @partial(jax.jit, static_argnames=("cls",))
    def empty_optimizers_states(
        cls,
        training_state: "PBTTD3TrainingState",
    ) -> "PBTTD3TrainingState":
        return training_state.replace(  # type: ignore
            target_critic_params=jnp.empty(shape=(1,), dtype=jnp.float32),
            target_policy_params=jnp.empty(shape=(1,), dtype=jnp.float32),
            policy_optimizer_state=jnp.empty(shape=(1,), dtype=jnp.float32),
            critic_optimizer_state=jnp.empty(shape=(1,), dtype=jnp.float32),
        )

    @classmethod
    @partial(jax.jit, static_argnames=("cls",))
    def resample_hyperparams(
        cls, training_state: "PBTTD3TrainingState"
    ) -> "PBTTD3TrainingState":

        random_key = training_state.random_key
        random_key, sub_key = jax.random.split(random_key)
        discount = jax.random.uniform(sub_key, shape=(), minval=0.9, maxval=1.0)

        random_key, sub_key = jax.random.split(random_key)
        policy_lr = jax.random.uniform(sub_key, shape=(), minval=3e-5, maxval=3e-3)

        random_key, sub_key = jax.random.split(random_key)
        critic_lr = jax.random.uniform(sub_key, shape=(), minval=3e-5, maxval=3e-3)

        random_key, sub_key = jax.random.split(random_key)
        noise_clip = jax.random.uniform(sub_key, shape=(), minval=0.0, maxval=1.0)

        random_key, sub_key = jax.random.split(random_key)
        policy_noise = jax.random.uniform(sub_key, shape=(), minval=0.0, maxval=1.0)

        random_key, sub_key = jax.random.split(random_key)
        expl_noise = jax.random.uniform(sub_key, shape=(), minval=0.0, maxval=0.2)

        return training_state.replace(  # type: ignore
            discount=discount,
            policy_lr=policy_lr,
            critic_lr=critic_lr,
            noise_clip=noise_clip,
            policy_noise=policy_noise,
            random_key=random_key,
            expl_noise=expl_noise,
        )


@dataclass
class PBTTD3Config:
    """Configuration for the PBT-TD3 algorithm"""

    episode_length: int = 1000
    batch_size: int = 256
    policy_delay: int = 2
    reward_scaling: float = 1.0
    soft_tau_update: float = 0.005
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    policy_hidden_layer_size: Tuple[int, ...] = (256, 256)


class PBTTD3(TD3):
    def __init__(self, config: PBTTD3Config, action_size: int):

        td3_config = TD3Config(
            episode_length=config.episode_length,
            batch_size=config.batch_size,
            policy_delay=config.policy_delay,
            reward_scaling=config.reward_scaling,
            soft_tau_update=config.soft_tau_update,
            critic_hidden_layer_size=config.critic_hidden_layer_size,
            policy_hidden_layer_size=config.policy_hidden_layer_size,
        )
        TD3.__init__(self, td3_config, action_size)

    def init(
        self, random_key: RNGKey, action_size: int, observation_size: int
    ) -> PBTTD3TrainingState:
        """Initialise the training state of the PBT-TD3 algorithm, through creation
        of optimizer states and params.

        Args:
            random_key: a random key used for random operations.
            action_size: the size of the action array needed to interact with the
                environment.
            observation_size: the size of the observation array retrieved from the
                environment.

        Returns:
            the initial training state.
        """

        training_state = TD3.init(self, random_key, action_size, observation_size)

        # Initial training state
        training_state = PBTTD3TrainingState(
            policy_optimizer_state=training_state.policy_optimizer_state,
            policy_params=training_state.policy_params,
            critic_optimizer_state=training_state.critic_optimizer_state,
            critic_params=training_state.critic_params,
            target_policy_params=training_state.target_policy_params,
            target_critic_params=training_state.target_critic_params,
            random_key=training_state.random_key,
            steps=training_state.steps,
            discount=None,
            policy_lr=None,
            critic_lr=None,
            noise_clip=None,
            policy_noise=None,
            expl_noise=None,
        )

        # Sample hyperparameters
        training_state = PBTTD3TrainingState.resample_hyperparams(training_state)

        return training_state  # type: ignore

    @partial(jax.jit, static_argnames=("self", "env", "deterministic"))
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
            training_state: the PBT-TD3 training state
            env: the environment
            deterministic: whether to select action in a deterministic way.
                Defaults to False.

        Returns:
            the new environment state
            the new PBT-TD3 training state
            the played transition
        """

        actions, random_key = self.select_action(
            obs=env_state.obs,
            policy_params=training_state.policy_params,
            random_key=training_state.random_key,
            expl_noise=training_state.expl_noise,
            deterministic=deterministic,
        )
        training_state = training_state.replace(
            random_key=random_key,
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

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        training_state: PBTTD3TrainingState,
        replay_buffer: ReplayBuffer,
    ) -> Tuple[PBTTD3TrainingState, ReplayBuffer, Metrics]:
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
        random_key = training_state.random_key
        samples, random_key = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )

        # Update Critic
        random_key, subkey = jax.random.split(random_key)
        critic_loss, critic_gradient = jax.value_and_grad(td3_critic_loss_fn)(
            training_state.critic_params,
            target_policy_params=training_state.target_policy_params,
            target_critic_params=training_state.target_critic_params,
            policy_fn=self._policy.apply,
            critic_fn=self._critic.apply,
            policy_noise=training_state.policy_noise,
            noise_clip=training_state.noise_clip,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            transitions=samples,
            random_key=subkey,
        )
        critic_optimizer = optax.adam(learning_rate=training_state.critic_lr)
        critic_updates, critic_optimizer_state = critic_optimizer.update(
            critic_gradient, training_state.critic_optimizer_state
        )
        critic_params = optax.apply_updates(
            training_state.critic_params, critic_updates
        )
        # Soft update of target critic network
        target_critic_params = jax.tree_util.tree_map(
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
            policy_optimizer = optax.adam(learning_rate=training_state.policy_lr)
            (policy_updates, policy_optimizer_state,) = policy_optimizer.update(
                policy_gradient, training_state.policy_optimizer_state
            )
            policy_params = optax.apply_updates(
                training_state.policy_params, policy_updates
            )
            # Soft update of target policy
            target_policy_params = jax.tree_util.tree_map(
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
            random_key=random_key,
            steps=training_state.steps + 1,
        )

        metrics = {
            "actor_loss": policy_loss,
            "critic_loss": critic_loss,
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
        ) -> Tuple[RNGKey, PBTTD3TrainingState, ReplayBuffer]:
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
            carry: Tuple[PBTTD3TrainingState, EnvState, ReplayBuffer],
            unused_arg: Any,
        ) -> Tuple[Tuple[PBTTD3TrainingState, EnvState, ReplayBuffer], Any]:
            (
                training_state,
                env_state,
                replay_buffer,
                metrics,
            ) = do_iteration(*carry)
            return (training_state, env_state, replay_buffer), metrics

        def train_fn(
            training_state: PBTTD3TrainingState,
            env_state: EnvState,
            replay_buffer: ReplayBuffer,
        ) -> Tuple[Tuple[PBTTD3TrainingState, EnvState, ReplayBuffer], Metrics]:
            (training_state, env_state, replay_buffer), metrics = jax.lax.scan(
                _scan_do_iteration,
                (training_state, env_state, replay_buffer),
                None,
                length=num_iterations,
            )
            return (training_state, env_state, replay_buffer), metrics

        return jax.vmap(train_fn)  # type: ignore
