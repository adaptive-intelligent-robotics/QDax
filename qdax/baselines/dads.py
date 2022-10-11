"""
A collection of functions and classes that define the algorithm Dynamics Aware Discovery
of Skills (DADS), ref: https://arxiv.org/abs/1907.01657.
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

from qdax.baselines.sac import SAC, SacConfig
from qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from qdax.core.neuroevolution.losses.dads_loss import make_dads_loss_fn
from qdax.core.neuroevolution.mdp_utils import TrainingState, get_first_episode
from qdax.core.neuroevolution.networks.dads_networks import make_dads_networks
from qdax.core.neuroevolution.normalization_utils import (
    RunningMeanStdState,
    normalize_with_rmstd,
    update_running_mean_std,
)
from qdax.core.neuroevolution.sac_utils import generate_unroll
from qdax.environments import CompletedEvalWrapper
from qdax.types import Metrics, Params, Reward, RNGKey, Skill, StateDescriptor


class DadsTrainingState(TrainingState):
    """Contains training state for the learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    critic_optimizer_state: optax.OptState
    critic_params: Params
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    target_critic_params: Params
    dynamics_optimizer_state: optax.OptState
    dynamics_params: Params
    random_key: RNGKey
    steps: jnp.ndarray
    normalization_running_stats: RunningMeanStdState


@dataclass
class DadsConfig(SacConfig):
    num_skills: int = 5
    dynamics_update_freq: int = 1
    descriptor_full_state: bool = False
    normalize_target: bool = True
    omit_input_dynamics_dim: int = 2


class DADS(SAC):
    """Implements DADS algorithm https://arxiv.org/abs/1907.01657.

    Note that the functions select_action, _update_alpha, _update_critic and
    _update_actor are inherited from SAC algorithm.

    In the current implementation, we suppose that the skills are fixed one
    hot vectors, and do not support continuous skills at the moment.

    Also, we suppose that the skills are evaluated in parallel in a fixed
    manner: a batch of environments, containing a multiple of the number
    of skills, is used to evaluate the skills in the environment and hence
    to generate transitions. The sampling is hence fixed and perfectly uniform.

    We plan to add continous skill as an option in the future. We also plan
    to release the current constraint on the number of batched environments
    by sampling from the skills rather than having this fixed setting.
    """

    def __init__(self, config: DadsConfig, action_size: int, descriptor_size: int):
        self._config: DadsConfig = config

        if self._config.normalize_observations:
            raise NotImplementedError("Normalization in not implemented for DADS yet")

        # define the networks
        self._policy, self._critic, self._dynamics = make_dads_networks(
            action_size=action_size,
            descriptor_size=descriptor_size,
            omit_input_dynamics_dim=config.omit_input_dynamics_dim,
        )

        # define the action distribution
        parametric_action_distribution = NormalTanhDistribution(event_size=action_size)
        self._sample_action_fn = parametric_action_distribution.sample

        # define the losses
        (
            self._alpha_loss_fn,
            self._policy_loss_fn,
            self._critic_loss_fn,
            self._dynamics_loss_fn,
        ) = make_dads_loss_fn(
            policy_fn=self._policy.apply,
            critic_fn=self._critic.apply,
            dynamics_fn=self._dynamics.apply,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            action_size=action_size,
            num_skills=self._config.num_skills,
            parametric_action_distribution=parametric_action_distribution,
        )

        # define the optimizers
        self._policy_optimizer = optax.adam(learning_rate=self._config.learning_rate)
        self._critic_optimizer = optax.adam(learning_rate=self._config.learning_rate)
        self._alpha_optimizer = optax.adam(learning_rate=self._config.learning_rate)
        self._dynamics_optimizer = optax.adam(learning_rate=self._config.learning_rate)

    def init(  # type: ignore
        self,
        random_key: RNGKey,
        action_size: int,
        observation_size: int,
        descriptor_size: int,
    ) -> DadsTrainingState:
        """Initialise the training state of the algorithm.

        Args:
            random_key: a jax random key
            action_size: the size of the environment's action space
            observation_size: the size of the environment's observation space
            descriptor_size: the size of the environment's descriptor space (i.e. the
                dimension of the dynamics network's input)

        Returns:
            the initial training state of DADS
        """
        # Initialize params
        dummy_obs = jnp.zeros((1, observation_size + self._config.num_skills))
        dummy_action = jnp.zeros((1, action_size))
        dummy_dyn_obs = jnp.zeros((1, descriptor_size))
        dummy_skill = jnp.zeros((1, self._config.num_skills))

        random_key, subkey = jax.random.split(random_key)
        policy_params = self._policy.init(subkey, dummy_obs)

        random_key, subkey = jax.random.split(random_key)
        critic_params = self._critic.init(subkey, dummy_obs, dummy_action)

        target_critic_params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x.copy()), critic_params
        )

        random_key, subkey = jax.random.split(random_key)
        dynamics_params = self._dynamics.init(
            subkey,
            obs=dummy_dyn_obs,
            skill=dummy_skill,
            target=dummy_dyn_obs,
        )

        policy_optimizer_state = self._policy_optimizer.init(policy_params)
        critic_optimizer_state = self._critic_optimizer.init(critic_params)
        dynamics_optimizer_state = self._dynamics_optimizer.init(dynamics_params)

        log_alpha = jnp.asarray(jnp.log(self._config.alpha_init), dtype=jnp.float32)
        alpha_optimizer_state = self._alpha_optimizer.init(log_alpha)

        return DadsTrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=log_alpha,
            target_critic_params=target_critic_params,
            dynamics_optimizer_state=dynamics_optimizer_state,
            dynamics_params=dynamics_params,
            random_key=random_key,
            normalization_running_stats=RunningMeanStdState(
                mean=jnp.zeros(
                    descriptor_size,
                ),
                var=jnp.ones(
                    descriptor_size,
                ),
                count=jnp.zeros(()),
            ),
            steps=jnp.array(0),
        )

    @partial(jax.jit, static_argnames=("self",))
    def _compute_diversity_reward(
        self, transition: QDTransition, training_state: DadsTrainingState
    ) -> Reward:
        """Computes the diversity reward of DADS.

        Args:
            transition: a batch of transitions from the replay buffer
            training_state: the current training state

        Returns:
            the diversity reward
        """
        active_skills = transition.obs[:, -self._config.num_skills :]

        # Compute dynamics prob
        next_state_desc = transition.next_state_desc
        state_desc = transition.state_desc
        target = next_state_desc - state_desc

        if self._config.normalize_target:
            target = normalize_with_rmstd(
                target, training_state.normalization_running_stats
            )

        log_q_phi = self._dynamics.apply(
            training_state.dynamics_params,
            state_desc,
            active_skills,
            target,
        )

        # Estimate prior skill
        skill_samples = jnp.tile(
            jnp.eye(self._config.num_skills), (state_desc.shape[0], 1)
        )
        state_descriptors = jnp.repeat(state_desc, self._config.num_skills, axis=0)
        target = jnp.repeat(target, self._config.num_skills, axis=0)
        log_p_s = self._dynamics.apply(
            training_state.dynamics_params,
            state_descriptors,
            skill_samples,
            target,
        )
        log_p_s = log_p_s.reshape((-1, self._config.num_skills))

        # Compute the reward according to DADS official implementation
        reward = jnp.log(self._config.num_skills) - jnp.log(
            jnp.exp(jnp.clip(log_p_s - log_q_phi.reshape((-1, 1)), -50, 50)).sum(axis=1)
        )

        return reward

    @partial(jax.jit, static_argnames=("self", "env", "deterministic", "evaluation"))
    def play_step_fn(
        self,
        env_state: EnvState,
        training_state: DadsTrainingState,
        env: Env,
        skills: Skill,
        deterministic: bool = False,
        evaluation: bool = False,
    ) -> Tuple[EnvState, DadsTrainingState, QDTransition]:
        """Plays a step in the environment. Concatenates skills to the observation
        vector, selects an action according to SAC rule and performs the environment
        step.

        Args:
            env_state: the current environment state
            training_state: the DIAYN training state
            skills: the skills concatenated to the observation vector
            env: the environment
            deterministic: the whether or not to select action in a deterministic way.
                Defaults to False.
            evaluation: if True, collected transitions are not used to update training
                state. Defaults to False.

        Returns:
            the new environment state
            the new DADS training state
            the played transition
        """

        random_key = training_state.random_key
        policy_params = training_state.policy_params
        obs = jnp.concatenate([env_state.obs, skills], axis=1)

        # If the env does not support state descriptor, we set it to (0,0)
        if "state_descriptor" in env_state.info:
            state_desc = env_state.info["state_descriptor"]
        else:
            state_desc = jnp.zeros((env_state.obs.shape[0], 2))

        actions, random_key = self.select_action(
            obs=obs,
            policy_params=policy_params,
            random_key=random_key,
            deterministic=deterministic,
        )

        next_env_state = env.step(env_state, actions)
        next_obs = jnp.concatenate([next_env_state.obs, skills], axis=1)
        if "state_descriptor" in next_env_state.info:
            next_state_desc = next_env_state.info["state_descriptor"]
        else:
            next_state_desc = jnp.zeros((next_env_state.obs.shape[0], 2))

        if self._config.normalize_target:
            if self._config.descriptor_full_state:
                _state_desc = obs[:, : -self._config.num_skills]
                _next_state_desc = next_obs[:, : -self._config.num_skills]
                target = _next_state_desc - _state_desc
            else:
                target = next_state_desc - state_desc

            target *= jnp.expand_dims(1 - next_env_state.done, -1)
            normalization_running_stats = update_running_mean_std(
                training_state.normalization_running_stats, target
            )
        else:
            normalization_running_stats = training_state.normalization_running_stats

        truncations = next_env_state.info["truncation"]
        transition = QDTransition(
            obs=obs,
            next_obs=next_obs,
            state_desc=state_desc,
            next_state_desc=next_state_desc,
            rewards=next_env_state.reward,
            dones=next_env_state.done,
            actions=actions,
            truncations=truncations,
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

        return next_env_state, training_state, transition

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "play_step_fn",
            "env_batch_size",
        ),
    )
    def eval_policy_fn(
        self,
        training_state: DadsTrainingState,
        eval_env_first_state: EnvState,
        play_step_fn: Callable[
            [EnvState, Params, RNGKey],
            Tuple[EnvState, Params, RNGKey, QDTransition],
        ],
        env_batch_size: int,
    ) -> Tuple[Reward, Reward, Reward, StateDescriptor]:
        """Evaluates the agent's policy over an entire episode, across all batched
        environments.


        Args:
            training_state: the DADS training state
            eval_env_first_state: the initial state for evaluation
            play_step_fn: the play_step function used to collect the evaluation episode
            env_batch_size: the number of environments we play simultaneously

        Returns:
            true return averaged over batch dimension, shape: (1,)
            true return per environment, shape: (env_batch_size,)
            diversity return per environment, shape: (env_batch_size,)
            state descriptors, shape: (episode_length, env_batch_size, descriptor_size)

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

        reshaped_transitions = jax.tree_util.tree_map(
            lambda x: x.reshape((self._config.episode_length * env_batch_size, -1)),
            transitions,
        )

        if self._config.descriptor_full_state:
            state_desc = reshaped_transitions.obs[:, : -self._config.num_skills]
            next_state_desc = reshaped_transitions.next_obs[
                :, : -self._config.num_skills
            ]
            reshaped_transitions = reshaped_transitions.replace(
                state_desc=state_desc, next_state_desc=next_state_desc
            )

        diversity_rewards = self._compute_diversity_reward(
            transition=reshaped_transitions, training_state=training_state
        ).reshape((self._config.episode_length, env_batch_size))

        diversity_returns = jnp.nansum(diversity_rewards, axis=0)

        return true_return, true_returns, diversity_returns, transitions.state_desc

    @partial(jax.jit, static_argnames=("self",))
    def _compute_reward(
        self, transition: QDTransition, training_state: DadsTrainingState
    ) -> Reward:
        """Computes the reward to train the networks.

        Args:
            transition: a batch of transitions from the replay buffer
            training_state: the current training state

        Returns:
            the DADS diversity reward
        """
        return self._compute_diversity_reward(
            transition=transition, training_state=training_state
        )

    @partial(jax.jit, static_argnames=("self",))
    def _update_dynamics(
        self, operand: Tuple[DadsTrainingState, QDTransition]
    ) -> Tuple[Params, float, optax.OptState]:
        """Update the dynamics network, independently of other networks. Called every
        `dynamics_update_freq` training steps.
        """
        training_state, transitions = operand

        dynamics_loss, dynamics_gradient = jax.value_and_grad(self._dynamics_loss_fn,)(
            training_state.dynamics_params,
            transitions,
        )

        (dynamics_updates, dynamics_optimizer_state,) = self._dynamics_optimizer.update(
            dynamics_gradient, training_state.dynamics_optimizer_state
        )
        dynamics_params = optax.apply_updates(
            training_state.dynamics_params, dynamics_updates
        )
        return (
            dynamics_params,
            dynamics_loss,
            dynamics_optimizer_state,
        )

    @partial(jax.jit, static_argnames=("self",))
    def _not_update_dynamics(
        self, operand: Tuple[DadsTrainingState, QDTransition]
    ) -> Tuple[Params, float, optax.OptState]:
        """Fake update of the dynamics, called every time we don't want to update
        the dynamics while we update the other networks.
        """

        training_state, _transitions = operand

        return (
            training_state.dynamics_params,
            jnp.nan,
            training_state.dynamics_optimizer_state,
        )

    @partial(jax.jit, static_argnames=("self",))
    def _update_networks(
        self,
        training_state: DadsTrainingState,
        transitions: QDTransition,
    ) -> Tuple[DadsTrainingState, Metrics]:
        """Updates the networks involved in DADS.

        Args:
            training_state: the current training state of the algorithm.
            transitions: transitions sampled from a replay buffer.
            random_key: a random key to handle stochasticity.

        Returns:
            The updated training state and training metrics.
        """

        random_key = training_state.random_key

        # Update skill-dynamics
        (dynamics_params, dynamics_loss, dynamics_optimizer_state,) = jax.lax.cond(
            training_state.steps % self._config.dynamics_update_freq == 0,
            self._update_dynamics,
            self._not_update_dynamics,
            (training_state, transitions),
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

        # Create new training state
        new_training_state = DadsTrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            target_critic_params=target_critic_params,
            dynamics_optimizer_state=dynamics_optimizer_state,
            dynamics_params=dynamics_params,
            random_key=random_key,
            normalization_running_stats=training_state.normalization_running_stats,
            steps=training_state.steps + 1,
        )
        metrics = {
            "actor_loss": policy_loss,
            "critic_loss": critic_loss,
            "dynamics_loss": dynamics_loss,
            "alpha_loss": alpha_loss,
            "alpha": alpha,
            "training_observed_reward_mean": jnp.mean(transitions.rewards),
            "target_mean": jnp.mean(transitions.next_state_desc),
            "target_std": jnp.std(transitions.next_state_desc),
        }

        return new_training_state, metrics

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        training_state: DadsTrainingState,
        replay_buffer: ReplayBuffer,
    ) -> Tuple[DadsTrainingState, ReplayBuffer, Metrics]:
        """Performs a training step to update the policy, the critic and the
        dynamics network parameters.

        Args:
            training_state: the current DADS training state
            replay_buffer: the replay buffer

        Returns:
            the updated DIAYN training state
            the replay buffer
            the training metrics
        """

        # Sample a batch of transitions in the buffer
        random_key = training_state.random_key
        transitions, random_key = replay_buffer.sample(
            random_key,
            sample_size=self._config.batch_size,
        )

        # Optionally replace the state descriptor by the observation
        if self._config.descriptor_full_state:
            _state_desc = transitions.obs[:, : -self._config.num_skills]
            _next_state_desc = transitions.next_obs[:, : -self._config.num_skills]
            transitions = transitions.replace(
                state_desc=_state_desc, next_state_desc=_next_state_desc
            )

        # Compute the reward
        rewards = self._compute_reward(
            transition=transitions, training_state=training_state
        )

        # Compute the target and optionally normalize it for the training
        if self._config.normalize_target:
            next_state_desc = normalize_with_rmstd(
                transitions.next_state_desc - transitions.state_desc,
                training_state.normalization_running_stats,
            )

        else:
            next_state_desc = transitions.next_state_desc - transitions.state_desc

        # Update the transitions
        transitions = transitions.replace(
            next_state_desc=next_state_desc, rewards=rewards
        )

        new_training_state, metrics = self._update_networks(
            training_state, transitions=transitions
        )

        return new_training_state, replay_buffer, metrics
