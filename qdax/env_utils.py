import functools
from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple, Dict

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import brax

from brax.envs.env import State as EnvState
from qdax.types import (
    Action,
    Descriptors,
    Done,
    Fitness,
    Observation,
    Params,
    Reward,
    RNGKey,
    StateDescriptor,
)


def training_inference(policy_model, params, obs):
    return policy_model.apply(params, obs)

training_inference_vec = jax.vmap(
        lambda *args: training_inference(*args),
        in_axes=(None, 0, 0),
        out_axes=0,
    )

def get_deterministic_actions(parameters):
    loc, scale = jnp.split(parameters, 2, axis=-1)
    act = jnp.tanh(loc)
    return act


@flax.struct.dataclass
class Transition:
    """Stores data corresponding to a transition in an env."""

    obs: Observation
    next_obs: Observation
    rewards: Reward
    dones: Done
    actions: Action
    state_desc: StateDescriptor

def play_step(
    env_state: EnvState,
    policy_params: Params,
    random_key: RNGKey,
    env: brax.envs.Env,
    policy_model: Any,
) -> Tuple[EnvState, Params, RNGKey, Transition]:
    """Play an environment step and return the updated state and the transition."""

    logits = training_inference(policy_model, policy_params, env_state.obs)
    # actions = parametric_action_distribution.sample(logits, key_sample)
    actions = get_deterministic_actions(logits)
    #actions = policy_model.apply(policy_params, env_state.obs)

    next_state = env.step(env_state, actions)

    transition = Transition(
        obs=env_state.obs,
        next_obs=next_state.obs,
        rewards=next_state.reward,
        dones=next_state.done,
        actions=actions,
        state_desc=env_state.info["state_descriptor"],
    )

    return next_state, policy_params, random_key, transition

@partial(jax.jit, static_argnames=("play_step_fn", "episode_length"))
def generate_unroll(
    init_state: EnvState,
    policy_params: Params,
    key: RNGKey,
    episode_length: int,
    play_step_fn: Callable[
        [EnvState, Params, RNGKey],
        Tuple[
            EnvState,
            Params,
            RNGKey,
            Transition,
        ],
    ],
) -> Tuple[EnvState, Transition]:
    """Generates an episode according to the agent's policy, returns the final state of the
    episode and the transitions of the episode.
    """

    def _scannable_play_step_fn(
        carry: Tuple[EnvState, Params, RNGKey], unused_arg: Any
    ) -> Tuple[Tuple[EnvState, Params, RNGKey,], Transition]:
        env_state, policy_params, random_key, transitions = play_step_fn(*carry)
        return (env_state, policy_params, random_key), transitions

    (state, _, _), transitions = jax.lax.scan(
        _scannable_play_step_fn,
        (init_state, policy_params, key),
        (),
        length=episode_length,
    )
    return state, transitions


# Define util functions
def play_step_vec(
    env_state: EnvState,
    policy_params: Params,
    random_key: RNGKey,
    env: brax.envs.Env,
    policy_model: Any,
) -> Tuple[EnvState, Params, RNGKey, Transition]:
    """Play an environment step and return the updated state and the transition."""

    logits = training_inference_vec(policy_model, policy_params, env_state.obs)
    # actions = parametric_action_distribution.sample(logits, key_sample)
    actions = get_deterministic_actions(logits)
    #actions = policy_model.apply(policy_params, env_state.obs)

    # this env state is already parallel
    next_state = env.step(env_state, actions)

    transition = Transition(
        obs=env_state.obs,
        next_obs=next_state.obs,
        rewards=next_state.reward,
        dones=next_state.done,
        actions=actions,
        state_desc=env_state.info["state_descriptor"],
    )

    return next_state, policy_params, random_key, transition











def init_run_eval(env_fn,
                  action_repeat: int,
                  batch_size: int,
                  episode_length: int = 1000,
                  seed: int = 0):
    """Initialize run_eval."""
    key_eval = jax.random.PRNGKey(seed=seed)
    core_eval_env = env_fn(
        action_repeat=action_repeat,
        batch_size=batch_size,
        episode_length=episode_length)
    jit_env_reset = jax.jit(core_eval_env.reset)
    jit_env_step = jax.jit(core_eval_env.step)
    first_state = jit_env_reset(key_eval)
    return core_eval_env, first_state, jit_env_step, jit_env_reset, key_eval

def jit_run_eval(env_fn,
                 inference_fn,
                 action_repeat: int,
                 batch_size: int,
                 episode_length: int = 1000,
                 seed: int = 0):
    """JIT run_eval."""
    core_eval_env, eval_first_state, eval_step_fn, jit_env_reset_fn, key_eval = init_run_eval(
        env_fn=env_fn,
        seed=seed,
        action_repeat=action_repeat,
        batch_size=batch_size,
        episode_length=episode_length)
    jit_inference_fn = jax.jit(inference_fn)

    def do_one_step_eval(carry, unused_target_t):
        state, params, key = carry
        key, key_sample = jax.random.split(key)
        actions = jit_inference_fn(params, state.core.obs, key_sample)
        nstate = eval_step_fn(state, actions)
        transition = Transition(
            obs=state.obs,
            next_obs=nstate.obs,
            rewards=nstate.reward,
            dones=nstate.done,
            actions=actions,
            state_desc=state.info["state_descriptor"],
        )
        return (nstate, params, key), (nstate.core, transition)

    @jax.jit
    def jit_run_eval_fn(state, key, params):
        (final_state, _, key), (states, transitions) = jax.lax.scan(
            do_one_step_eval, (state, params, key), (),
            length=episode_length // action_repeat)
        return final_state, key, states, transitions

    return core_eval_env, eval_first_state, key_eval, jit_run_eval_fn, jit_env_reset_fn



def rollout_env(
    params: Dict[str, Dict[str, jnp.ndarray]],
    env_fn,
    inference_fn,
    batch_size: int = 0,
    seed: int = 0,
    reset_args: Tuple[Any] = (),
    step_args: Tuple[Any] = (),
    step_fn_name: str = 'step',
):
    """Visualize environment."""
    rng = jax.random.PRNGKey(seed=seed)
    rng, reset_key = jax.random.split(rng)
    env = env_fn(batch_size=batch_size)
    inference_fn = inference_fn or functools.partial(
        training_utils.zero_fn, action_size=env.action_size)
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(getattr(env, step_fn_name))
    jit_inference_fn = jax.jit(inference_fn)
    states = []
    state = jit_env_reset(reset_key, *reset_args)
    while not jnp.all(state.done):
        states.append(state)
        tmp_key, rng = jax.random.split(rng)
        act = jit_inference_fn(params, state.obs, tmp_key)
        next_state = jit_env_step(state, act, *step_args)

        transition = Transition(
            obs=state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=act,
            state_desc=state.info["state_descriptor"],
        )

    states.append(next_state)

    return env, states
