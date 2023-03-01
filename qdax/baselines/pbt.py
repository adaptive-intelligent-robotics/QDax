from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode

from qdax.core.neuroevolution.buffers.buffer import ReplayBuffer
from qdax.types import RNGKey


class PBTTrainingState(PyTreeNode):
    """
    The state of a  PBT training process. Can be used to store anything
    that is useful for a training process. This object is used in the
    package to store all stateful object necessary for training an agent
    that learns how to act in an MDP. This class should be used as base
    class for inheritance for any algorithm that want to implement a PBT
    scheme.
    """

    pass

    @classmethod
    @partial(jax.jit, static_argnames=("cls",))
    def resample_hyperparams(
        cls, training_state: "PBTTrainingState"
    ) -> "PBTTrainingState":
        """
        Resample the agents hyperparameters.
        """
        raise NotImplementedError()

    @classmethod
    @partial(jax.jit, static_argnames=("cls",))
    def init_optimizers_states(
        cls,
        training_state: "PBTTrainingState",
    ) -> "PBTTrainingState":
        """
        Initialize the agents optimizers states.
        """
        raise NotImplementedError()

    @classmethod
    @partial(jax.jit, static_argnames=("cls",))
    def empty_optimizers_states(
        cls,
        training_state: "PBTTrainingState",
    ) -> "PBTTrainingState":
        """
        Empty the agents optimizers states. Might be useful in some settings
        to save memory usage.
        """
        raise NotImplementedError()


class PBT:
    """
    This class serves as a template for algorithm that want to implement the standard
    Population Based Training (PBT) scheme.
    """

    def __init__(
        self,
        population_size: int,
        num_best_to_replace_from: int,
        num_worse_to_replace: int,
    ):
        """

        Args:
            population_size: Size of the PBT population.
            num_best_to_replace_from: Number of top performing individuals to sample
                from when replacing low performers at each iteration.
            num_worse_to_replace: Number of low-performing individuals to replace at
                each iteration.
        """
        if num_best_to_replace_from + num_worse_to_replace > population_size:
            raise ValueError(
                "The sum of best number of individuals to replace "
                "from and worse individuals to replace exceeds the population size."
            )
        self._population_size = population_size
        self._num_best_to_replace_from = num_best_to_replace_from
        self._num_worse_to_replace = num_worse_to_replace

    @partial(jax.jit, static_argnames=("self",))
    def update_states_and_buffer(
        self,
        random_key: RNGKey,
        population_returns: jnp.ndarray,
        training_state: PBTTrainingState,
        replay_buffer: ReplayBuffer,
    ) -> Tuple[RNGKey, PBTTrainingState, ReplayBuffer]:
        """
        Updates the agents of the population states as well as
        their shared replay buffer.

        Args:
            random_key: Random RNG key.
            population_returns: Returns of the agents in the populations.
            training_state: The training state of the PBT scheme.
            replay_buffer: Shared replay buffer by the agents.

        Returns:
            Updated random key, updated PBT training state and updated replay buffer.
        """
        indices_sorted = jax.numpy.argsort(-population_returns)
        best_indices = indices_sorted[: self._num_best_to_replace_from]
        indices_to_replace = indices_sorted[-self._num_worse_to_replace :]

        random_key, key = jax.random.split(random_key)
        indices_used_to_replace = jax.random.choice(
            key, best_indices, shape=(self._num_worse_to_replace,), replace=True
        )

        training_state = jax.tree_util.tree_map(
            lambda x, y: x.at[indices_to_replace].set(y[indices_used_to_replace]),
            training_state,
            jax.vmap(training_state.__class__.resample_hyperparams)(training_state),
        )

        replay_buffer = jax.tree_util.tree_map(
            lambda x, y: x.at[indices_to_replace].set(y[indices_used_to_replace]),
            replay_buffer,
            replay_buffer,
        )

        return random_key, training_state, replay_buffer

    @partial(jax.jit, static_argnames=("self",))
    def update_states_and_buffer_pmap(
        self,
        random_key: RNGKey,
        population_returns: jnp.ndarray,
        training_state: PBTTrainingState,
        replay_buffer: ReplayBuffer,
    ) -> Tuple[RNGKey, PBTTrainingState, ReplayBuffer]:
        """
        Updates the agents of the population states as well as
        their shared replay buffer. This is the version of the function to be
        used within jax.pmap. It makes the population is spread over several devices
        and implement a parallel update through communication between the devices.

        Args:
            random_key: Random RNG key.
            population_returns: Returns of the agents in the populations.
            training_state: The training state of the PBT scheme.
            replay_buffer: Shared replay buffer by the agents.

        Returns:
            Updated random key, updated PBT training state and updated replay buffer.
        """
        indices_sorted = jax.numpy.argsort(-population_returns)
        best_indices = indices_sorted[: self._num_best_to_replace_from]
        indices_to_replace = indices_sorted[-self._num_worse_to_replace :]

        best_individuals, best_buffers, best_returns = jax.tree_util.tree_map(
            lambda x: x[best_indices],
            (training_state, replay_buffer, population_returns),
        )
        (
            gathered_best_individuals,
            gathered_best_buffers,
            gathered_best_returns,
        ) = jax.tree_util.tree_map(
            lambda x: jnp.concatenate(jax.lax.all_gather(x, axis_name="p"), axis=0),
            (best_individuals, best_buffers, best_returns),
        )
        pop_indices_sorted = jax.numpy.argsort(-gathered_best_returns)
        best_pop_indices = pop_indices_sorted[: self._num_best_to_replace_from]

        random_key, key = jax.random.split(random_key)
        indices_used_to_replace = jax.random.choice(
            key, best_pop_indices, shape=(self._num_worse_to_replace,), replace=True
        )

        training_state = jax.tree_util.tree_map(
            lambda x, y: x.at[indices_to_replace].set(y[indices_used_to_replace]),
            training_state,
            jax.vmap(gathered_best_individuals.__class__.resample_hyperparams)(
                gathered_best_individuals
            ),
        )

        replay_buffer = jax.tree_util.tree_map(
            lambda x, y: x.at[indices_to_replace].set(y[indices_used_to_replace]),
            replay_buffer,
            gathered_best_buffers,
        )

        return random_key, training_state, replay_buffer
