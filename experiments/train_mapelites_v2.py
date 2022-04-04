"""Launching the MAP-Elites algorithm on a Brax environment.
See: https://arxiv.org/abs/1504.04909
"""

import dataclasses
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Sequence, Tuple

import brax
import flax
import jax
import jax.numpy as jnp
from brax.envs.env import State as EnvState
from jax.config import config
from jax.flatten_util import ravel_pytree
from qdax import brax_envs
from qdax.algorithms.map_elites import MAPElites, QDMetrics, compute_cvt_centroids
from qdax.algorithms.mutation_operators import (
    isoline_crossover_function,
    polynomial_mutation_function,
)
from qdax.algorithms.plotting import plot_2d_map_elites_grid
from qdax.algorithms.types import (
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
from qdax.algorithms.utils import MLP, generate_unroll

config.update("jax_disable_jit", False)


@flax.struct.dataclass
class Transition:
    """Stores data corresponding to a transition in an env."""

    obs: Observation
    next_obs: Observation
    rewards: Reward
    dones: Done
    actions: Action
    state_desc: StateDescriptor


def flatten_policy_variables(policy_params: Params) -> jnp.ndarray:
    flatten_variables, _ = ravel_pytree(policy_params)
    return flatten_variables


# Define util functions
def play_step(
    env_state: EnvState,
    policy_params: Params,
    random_key: RNGKey,
    env: brax.envs.Env,
) -> Tuple[EnvState, Params, RNGKey, Transition]:
    """Play an environment step and return the updated state and the transition."""

    actions = jax.vmap(policy_network.apply)(policy_params, env_state.obs)

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


def get_final_xy_position(data: Transition, mask: jnp.ndarray) -> Descriptors:
    """Compute final xy positon.

    This function suppose that state descriptor is the xy position, as it
    just select the final one of the state descriptors given.
    """
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    # Get behavior descriptor
    last_index = jnp.int32(jnp.sum(1.0 - mask, axis=0)) - 1

    descriptors = jax.vmap(lambda x, y: x[y], in_axes=(1, 0))(
        data.state_desc, last_index
    )

    return descriptors.squeeze()


def get_feet_contact_proportion(data: Transition, mask: jnp.ndarray) -> Descriptors:
    """Compute feet contact time proportion.

    This function suppose that state descriptor is the feet contact, as it
    just computes the mean of the state descriptors given.
    """
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    # Get behavior descriptor
    descriptors = jnp.sum(data.state_desc * (1.0 - mask), axis=0)
    descriptors = descriptors / jnp.sum(1.0 - mask, axis=0)

    return descriptors


def scoring_function(
    flatten_variables: Params,
    init_state: brax.envs.State,
    episode_length: int,
    random_key: RNGKey,
    play_step_fn: Callable,
    behavior_descriptor_extractor: Callable,
) -> Tuple[Fitness, Descriptors]:
    """Evaluate policies contained in flatten_variables in parallel"""

    policies_params = jax.vmap(policy_recontruction_fn)(flatten_variables)

    # Perform rollouts with each policy
    unroll_fn = partial(
        generate_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        key=random_key,
    )

    _final_state, data = unroll_fn(init_state, policies_params)

    # create a mask to extract data properly
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=0), 0, 1)
    mask = jnp.roll(is_done, 1, axis=0)
    mask = mask.at[0, :].set(0)

    # Scores
    fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=0)

    descriptors = behavior_descriptor_extractor(data, mask)

    return fitnesses, descriptors


@dataclass
class ExpConfig:
    """Configuration from this experiment script"""

    env_name: str = "pointmaze"
    seed: int = 0
    batch_size: int = 1000
    num_iterations: int = 100
    episode_length: int = 200
    num_centroids: int = 1000
    num_init_cvt_samples: int = 50000
    crossover_percentage: float = 0.5
    policy_hidden_layer_sizes: Sequence[int] = dataclasses.field(default_factory=list)
    # others
    log_period: int = 10
    _alg_name: str = "MAP-Elites"


if __name__ == "__main__":

    config = ExpConfig(
        env_name="pointmaze",
        seed=0,
        batch_size=2000,
        num_iterations=1000,
        episode_length=200,
        num_centroids=1000,
        num_init_cvt_samples=50000,
        crossover_percentage=0.5,
        policy_hidden_layer_sizes=[256, 256],
        # others
        log_period=100,
        _alg_name="MAP-Elites",
    )

    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")

    exp_date = datetime.today().strftime("%Y-%m-%d-%H:%M:%S")

    new_dir = (
        current_dir
        + "/"
        + "exp_outputs/"
        + config._alg_name
        + "/"
        + config.env_name
        + "/"
        + exp_date
        + "/"
    )

    # Change the current working directory
    os.makedirs(new_dir)
    os.chdir(new_dir)

    print(f"New working dir : {os.getcwd()}")

    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().handlers[0].setLevel(logging.INFO)
    logger = logging.getLogger(f"{__name__}")

    if not Path("images").is_dir():
        try:
            os.mkdir("images")
            os.mkdir("images/me_grids")
            os.mkdir("last_grid")
        except Exception:
            pass

    # Define components of the algorithm

    # Init environment
    env_name = config.env_name
    env = brax_envs.create(env_name, batch_size=config.batch_size)

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init policy network
    policy_layer_sizes = config.policy_hidden_layer_sizes + [env.action_size]
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        kernel_init_final=jax.nn.initializers.uniform(1e-3),
        final_activation=jnp.tanh,
    )

    # Init population of policies - and reconstruction function
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=config.batch_size)
    fake_batch = jnp.zeros(shape=(config.batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)
    init_flatten_variables = jax.vmap(flatten_policy_variables)(init_variables)
    _, policy_recontruction_fn = ravel_pytree(
        jax.tree_map(lambda x: x[0], init_variables)
    )

    play_step_fn = jax.jit(partial(play_step, env=env))

    # makes evaluation fully determinist
    random_key, subkey = jax.random.split(random_key)
    reset_fn = jax.jit(env.reset)
    init_state = reset_fn(subkey)

    env_bd_extractor = {
        "pointmaze": get_final_xy_position,
        "ant_omni": get_final_xy_position,
        "walker2d_uni": get_feet_contact_proportion,
    }

    assert (
        env_name in env_bd_extractor.keys()
    ), "Please register the bd extractor needed"

    # prepare and jit the scoring function
    scoring_fn = jax.jit(
        partial(
            scoring_function,
            init_state=init_state,
            episode_length=config.episode_length,
            random_key=random_key,
            play_step_fn=play_step_fn,
            behavior_descriptor_extractor=env_bd_extractor[env_name],
        )
    )

    # Define mutation function and crossover function
    mutation_fn = jax.jit(
        partial(
            polynomial_mutation_function,
            proportion_to_mutate=0.1,
            eta=0.05,
            minval=-1.0,
            maxval=1.0,
        )
    )

    crossover_fn = partial(
        isoline_crossover_function,
        iso_sigma=0.005,
        line_sigma=0.05,
    )

    # Beginning of the algorithm

    # create the mapelites instance
    mapelites = MAPElites(
        scoring_function=scoring_fn,
        crossover_function=crossover_fn,
        mutation_function=mutation_fn,
        crossover_percentage=config.crossover_percentage,
    )

    # compute the centroids
    logger.warning("--- Compute the CVT centroids ---")
    minval, maxval = env.behavior_descriptor_limits
    init_time = time.time()
    centroids = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=config.num_init_cvt_samples,
        num_centroids=config.num_centroids,
        minval=minval,
        maxval=maxval,
    )

    duration = time.time() - init_time
    logger.warning(f"--- Duration for CVT centroids computation : {duration:.2f}s")

    # jit functions
    init_function = partial(
        mapelites.init_fn,
        centroids=centroids,
    )
    init_function = jax.jit(init_function)

    update_fn = partial(mapelites.update_fn, batch_size=config.batch_size)

    @jax.jit
    def iteration_fn(carry, unused):
        # iterate over grid
        grid, rkey = carry
        (grid, rkey,) = update_fn(
            grid,
            rkey,
        )

        # get metrics
        grid_empty = grid.grid_scores == -jnp.inf
        qd_score = jnp.sum(grid.grid_scores, where=~grid_empty)
        coverage = 100 * jnp.mean(1.0 - grid_empty)
        max_fitness = jnp.max(grid.grid_scores)

        return (
            grid,
            rkey,
        ), QDMetrics(qd_score, max_fitness, coverage)

    # init algorithm
    logger.warning("--- Algorithm initialisation ---")
    grid = init_function(init_flatten_variables)
    logger.warning("--- Initialised ---")

    logger.warning("--- Starting the algorithm main process ---")

    current_step_estimation = 0
    total_training_time = 0

    # Main loop
    num_loop_iterations = config.num_iterations // config.log_period
    for iteration in range(num_loop_iterations):

        logger.warning(
            f"--- Iteration indice : {iteration} out of {num_loop_iterations} ---"
        )
        logger.warning(
            f"--- Iteration number : {iteration * config.log_period}"
            f" out of {num_loop_iterations * config.log_period}---"
        )

        start_time = time.time()
        (grid, random_key,), metrics = jax.lax.scan(
            iteration_fn,
            (
                grid,
                random_key,
            ),
            (),
            length=config.log_period,
        )
        time_duration = time.time() - start_time
        total_training_time += time_duration

        # update nb steps estimation
        current_step_estimation += (
            config.batch_size * config.episode_length * config.log_period
        )

        logger.warning(
            f"--- Current nb steps in env (estimation): {current_step_estimation:.2f}"
        )
        logger.warning(f"--- Time duration (batch of iteration): {time_duration:.2f}")
        logger.warning(
            f"--- Time duration (Sum of iterations): {total_training_time:.2f}"
        )
        logger.warning(f"--- Current QD Score: {metrics.qd_score[-1]:.2f}")
        logger.warning(f"--- Current Coverage: {metrics.coverage[-1]:.2f}%")
        logger.warning(f"--- Current Max Fitness: {metrics.max_fitness[-1]}")

        if env.behavior_descriptor_length == 2:
            plot_2d_map_elites_grid(
                centroids=centroids,
                grid_fitness=grid.grid_scores,
                minval=minval,
                maxval=maxval,
                grid_descriptors=grid.grid_descriptors,
                ax=None,
                save_to_path=f"images/me_grids/grid_{current_step_estimation}",
            )

        # store the latest controllers
        store_entire_me_grid = True
        if store_entire_me_grid:
            grid.save(path="last_grid/")

    duration = time.time() - init_time

    logger.warning("--- Final metrics ---")

    logger.warning(f"Duration: {duration:.2f}s")
    logger.warning(f"Training duration: {total_training_time:.2f}s")
    logger.warning(f"QD Score: {metrics.qd_score[-1]:.2f}")
    logger.warning(f"Coverage: {metrics.coverage[-1]:.2f}%")
