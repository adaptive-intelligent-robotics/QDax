import abc
import functools
import logging
import time
from typing import Callable, Any, Optional

import jax
import jax.numpy as jnp
import flax.struct

from brax import envs
from brax.envs import wrappers
from brax.training import distribution, networks
from qdax.envs import deterministic_ant_cost_reward
from qdax.stats.training_state import TrainingState
from qdax.envs.unidirectional_envs import ant, hopper, walker
from qdax.envs.unidirectional_envs import ant, walker, hopper, halfcheetah, humanoid
from qdax.envs.omnidirectional_envs import ant as ant_omni
from qdax.envs.omnidirectional_envs import humanoid as humanoid_omni


class BraxTask(object):
  def __init__(self,
               env_name: str,
               episode_length: int,
               action_repeat: int,
               num_envs: int,
               local_devices_to_use: int,
               process_count: int):

    self.core_env = BraxTask.generate_env(
      env_name=env_name,
      action_repeat=action_repeat,
      batch_size=num_envs // local_devices_to_use // process_count,
      episode_length=episode_length
    )

    self.step_fn = jax.jit(
      self.core_env.step
    )

    self.reset_fn = jax.jit(
      self.core_env.reset
    )

    self.reset_fn_batch = jax.jit(
      jax.vmap(
        self.core_env.reset
      )
    )

    self.policy_model = BraxTask._generate_individual_model(
      observation_size=self.core_env.observation_size,
      action_size=self.core_env.action_size
    )

    self.eval = BraxTask._create_eval_function(
      self.policy_model,
      self.reset_fn,
      self.step_fn,
      episode_length,
      action_repeat,
    )

    self.get_random_parameters = BraxTask._create_random_parameters_function(self.policy_model)

    self.get_bd = BraxTask._create_get_bd_fn()

  @staticmethod
  def _create_get_bd_fn():
    return jax.jit(
      BraxTask._get_bd
    )

  @staticmethod
  def _create_eval_function(policy_model, reset_fn, step_fn, episode_length, action_repeat):
    return functools.partial(
      BraxTask._eval,
      policy_model,
      reset_fn,
      step_fn,
      episode_length,
      action_repeat,
    )

  @staticmethod
  def _create_random_parameters_function(policy_model):
    return functools.partial(
      BraxTask.get_random_parameters,
      policy_model,
    )

  @staticmethod
  def generate_env(env_name: str,
                   episode_length: int = 100,
                   action_repeat: int = 1,
                   auto_reset: bool = True,
                   batch_size: Optional[int] = None,
                   **kwargs):
    """Creates an Env with a specified brax system."""
    env = {'ant': ant.QDUniAnt(**kwargs),
     'hopper': hopper.QDUniHopper(**kwargs),
     'walker': walker.QDUniWalker(**kwargs),
     'halfcheetah': halfcheetah.QDUniHalfcheetah(**kwargs),
     'humanoid': humanoid.QDUniHumanoid(**kwargs),
     'ant_omni': ant_omni.QDOmniAnt(**kwargs),
     'humanoid_omni': humanoid_omni.QDOmniHumanoid(**kwargs),
     }[env_name]

    if episode_length is not None:
      env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
      env = wrappers.VectorWrapper(env, batch_size)
    if auto_reset:
      env = wrappers.AutoResetWrapper(env)

    return env

  @staticmethod
  def _generate_individual_model(observation_size, action_size):
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    return networks.make_model(
      [64, 64, parametric_action_distribution.param_size],
      observation_size,
    )

  @staticmethod
  def _do_one_step(policy_model, step_fn, carry, unused_target_t):
    state, policy_params, key, cumulative_reward, active_episode = carry
    key, key_sample = jax.random.split(key)
    obs = state.obs
    logits = BraxTask.training_inference(policy_model, policy_params, obs)
    # actions = parametric_action_distribution.sample(logits, key_sample)
    actions = BraxTask._get_deterministic_actions(logits)
    nstate = step_fn(state, actions)
    cumulative_reward = cumulative_reward + nstate.reward * active_episode
    new_active_episode = active_episode * (1 - nstate.done)
    return (nstate, policy_params, key, cumulative_reward, new_active_episode), \
           (state.obs, active_episode, state.done, state.info['bd'], cumulative_reward)

  @staticmethod
  def _training_inference(policy_model, params, obs):
    return policy_model.apply(params, obs)

  training_inference = jax.vmap(
    lambda *args: BraxTask._training_inference(*args),
    in_axes=(None, 0, 0),
    out_axes=0,
  )

  @staticmethod
  def _get_deterministic_actions(parameters):
    loc, scale = jnp.split(parameters, 2, axis=-1)
    act = jnp.tanh(loc)
    return act

  @staticmethod
  def get_random_parameters(policy_model,
                            training_state: TrainingState,
                            population_size: int,
                            key_model
                            ):
    t = time.time()
    key_model_batch = jax.random.split(key_model, population_size)
    policy_params_fun = jax.vmap(policy_model.init)
    pparams = policy_params_fun( key_model_batch)
    logging.info("Init policies  %s ", time.time() - t)
    return pparams

  @staticmethod
  def _eval(policy_model,
            reset_fn,
            step_fn,
            episode_length,
            action_repeat,
            state: envs.State,
            params,
            key
            ):
    do_one_step_fn = functools.partial(BraxTask._do_one_step, policy_model, step_fn)

    cumulative_reward = jnp.zeros(state.obs.shape[0])
    active_episode = jnp.ones_like(cumulative_reward)
    key, key_env = jax.random.split(key, 2)
    state = reset_fn(key_env)
    (state, _, key, cumulative_reward, _), (obs, obs_weights, done, bd, reward_trajectory) = jax.lax.scan(
      # I don't understand why obs_weights is called "active_episode" in the function above.
      do_one_step_fn, (state, params, key, cumulative_reward, active_episode), (),
      length=episode_length // action_repeat)
    return cumulative_reward, obs, state, done, bd, reward_trajectory

  @staticmethod
  def _get_bd(obs_traj, first_done):
    # Selecting only the required dimensions of the observations but keeping all the timesteps
    bds = obs_traj[0, :, :, :]
    bds = jnp.moveaxis(bds, -1, 0)
    # bds = bds.reshape(-1, bds.shape[0], bds.shape[1]) # not sure if reshape does the right thing, gives correct shape but does it mess up the data??

    # Selecting only the wanted timestep
    bds = jnp.take_along_axis(bds, first_done.reshape(1, 1, len(first_done)), axis=1)
    bds = bds.reshape(bds.shape[0], -1)  # Removing the unnecessary axis

    return bds
