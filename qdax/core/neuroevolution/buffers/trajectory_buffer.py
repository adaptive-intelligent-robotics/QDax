from __future__ import annotations

from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from qdax.core.neuroevolution.buffers.buffer import Transition
from qdax.types import Reward, RNGKey


class TrajectoryBuffer(struct.PyTreeNode):
    """
    A buffer that stores transitions in the form of trajectories. Like `FlatBuffer`
    transitions are flattened before being stored and unflattened on the fly and the
    data shape is: (buffer_size, transition_concat_shape).
    The speicificity lies in the additional episodic data buffer that maps transitions
    that belong to the same trajectory to their position in the main buffer.

    Example:
    Assume we have a buffer of size 6, we insert 3 transitions at a time
    (env_batch_size=3) and the episode length is 2.
    The `dones` data we insert is dones=[0,1,0].

        Data (dones):
            [ 0.  1.  0. nan nan nan] # We inserted [0,1,0] contiguously
        Episodic data:
            [[ 0. nan] # For episode 0, first timestep is at index 0 in the buffer
            [ 1. nan]  # For episode 1, first timestep is at index 1 in the buffer
            [ 2. nan]] # For episode 2, first timestep is at index 2 in the buffer
        Trajectory positions:
            [0. 1. 0.] # For episode 0 and 2, done=0 so we stay in the same episode,
                       # for episode 1, done=1 so we move to the next episode index
        Timestep positions:
            [1. 0. 1.] # For episode 0 and 2: done=0 so we increment the timestep count-
                       # er, for episode 1: done=1 so we reset the timestep counter


    Now we subsequently add dones=[1,1,1]
        Data (dones):
            [0. 1. 0. 1. 1. 1.]
        Episodic data:
            [[ 0.  3.]
            [ 4. nan] # Episode 1 has been reset
            [ 2.  5.]]
        Trajectory positions:
            [1. 2. 1.]
        Timestep positions:
            [0. 0. 0.] # All timestep counters have been reset
    """

    data: jnp.ndarray
    buffer_size: int = struct.field(pytree_node=False)
    transition: Transition

    episode_length: int = struct.field(pytree_node=False)
    env_batch_size: int = struct.field(pytree_node=False)
    num_trajectories: int = struct.field(pytree_node=False)

    current_position: jnp.ndarray = struct.field()
    current_size: jnp.ndarray = struct.field()
    trajectory_positions: jnp.ndarray = struct.field()
    timestep_positions: jnp.ndarray = struct.field()
    episodic_data: jnp.ndarray = struct.field()
    current_episodic_data_size: jnp.ndarray = struct.field()
    returns: jnp.ndarray = struct.field()

    @classmethod
    def init(  # type: ignore
        cls,
        buffer_size: int,
        transition: Transition,
        env_batch_size: int,
        episode_length: int,
    ) -> TrajectoryBuffer:
        """
        The constructor of the buffer.

        Note: We have to define a classmethod instead of just doing it in post_init
        because post_init is called every time the dataclass is tree_mapped. This is a
        workaround proposed in https://github.com/google/flax/issues/1628.
        """
        flatten_dim = transition.flatten_dim
        data = jnp.ones((buffer_size, flatten_dim)) * jnp.nan
        num_trajectories = buffer_size // episode_length
        assert (
            num_trajectories % env_batch_size == 0
        ), "num_trajectories must be a multiple of env batch size"
        assert (
            buffer_size % episode_length == 0
        ), "buffer_size must be a multiple of episode_length"
        current_position = jnp.zeros((), dtype=int)
        trajectory_positions = jnp.zeros(env_batch_size, dtype=int)
        timestep_positions = jnp.zeros(env_batch_size, dtype=int)
        episodic_data = jnp.ones((num_trajectories, episode_length)) * jnp.nan
        current_size = jnp.array(0, dtype=int)
        current_episodic_data_size = jnp.array(0, dtype=int)
        returns = jnp.ones(
            buffer_size + 1,
        ) * (-jnp.inf)
        return cls(
            data=data,
            current_position=current_position,
            buffer_size=buffer_size,
            timestep_positions=timestep_positions,
            trajectory_positions=trajectory_positions,
            episode_length=episode_length,
            env_batch_size=env_batch_size,
            episodic_data=episodic_data,
            num_trajectories=num_trajectories,
            current_size=current_size,
            current_episodic_data_size=current_episodic_data_size,
            transition=transition,
            returns=returns,
        )

    @partial(jax.jit, static_argnames=("sample_size"))
    def sample(
        self,
        random_key: RNGKey,
        sample_size: int,
    ) -> Tuple[Transition, RNGKey]:
        """
        Sample transitions from the buffer. If sample_traj=False, returns stacked
        transitions in the shape (sample_size,), if sample_traj=True, return transitions
        in the shape (sample_size, episode_length,).
        """

        # Here we want to sample single transitions
        # We sample uniformly at random the indexes of valid transitions
        random_key, subkey = jax.random.split(random_key)
        idx = jax.random.randint(
            subkey,
            shape=(sample_size,),
            minval=0,
            maxval=self.current_size,
        )
        # We select the corresponding transitions
        samples = jnp.take(self.data, idx, axis=0, mode="clip")

        # (sample_size, concat_dim)
        transitions = self.transition.__class__.from_flatten(samples, self.transition)
        return transitions, random_key

    def sample_with_returns(
        self,
        random_key: RNGKey,
        sample_size: int,
    ) -> Tuple[Transition, Reward, RNGKey]:
        """Sample transitions and the return corresponding to their episode. The returns
        are compute by the method `compute_returns`.

        Args:
            random_key: a random key
            sample_size: the number of transitions

        Returns:
            The transitions, the associated returns and a new random key.
        """
        # Here we want to sample single transitions
        # We sample uniformly at random the indexes of valid transitions
        random_key, subkey = jax.random.split(random_key)
        idx = jax.random.randint(
            subkey,
            shape=(sample_size,),
            minval=0,
            maxval=self.current_size,
        )
        # We select the corresponding transitions
        samples = jnp.take(self.data, idx, axis=0, mode="clip")
        returns = jnp.take(self.returns, idx, mode="clip")
        # (sample_size, concat_dim)
        transitions = self.transition.__class__.from_flatten(samples, self.transition)
        return transitions, returns, random_key

    @jax.jit
    def insert(self, transitions: Transition) -> TrajectoryBuffer:
        """
        Scan over 'insert_one_transition', to add multiple transitions.
        """

        @jax.jit
        def insert_one_transition(
            replay_buffer: TrajectoryBuffer, flattened_transitions: jnp.ndarray
        ) -> Tuple[TrajectoryBuffer, Any]:
            """
            Insert a batch (one step over all envs) of transitions in the buffer.
            """
            # Step 1: reset episodes for override
            # We start by selecting the episodes that are currently being inserted
            active_trajectories_indexes = (
                jnp.arange(self.env_batch_size, dtype=int)
                + (replay_buffer.trajectory_positions % self.num_trajectories)
                * self.env_batch_size
            ) % self.num_trajectories

            current_episodes = replay_buffer.episodic_data[active_trajectories_indexes]

            # The override condition is: "if we want to insert à timestep 0, we clear
            # the corresponding row first"
            condition = replay_buffer.timestep_positions % self.episode_length == 0

            # Clear episodes that match the condition, don't modify others
            override_episodes = jnp.where(
                jnp.expand_dims(condition, -1),
                jnp.ones_like(current_episodes) * jnp.nan,
                current_episodes,
            )

            new_episodic_data = replay_buffer.episodic_data.at[
                active_trajectories_indexes
            ].set(override_episodes)

            # Step 2: insert data in main buffer and indexes in episodic buffer
            # Apply changes in the episodic_data array

            # Insert transitions in the buffer
            new_data = jax.lax.dynamic_update_slice_in_dim(
                replay_buffer.data,
                flattened_transitions,
                start_index=replay_buffer.current_position % self.buffer_size,
                axis=0,
            )

            # We inserted from current_position to current_position + env_batch_size
            inserted_indexes = (
                jnp.arange(
                    self.env_batch_size,
                )
                + replay_buffer.current_position
            )
            # We set the indexes of inserted episodes in the episodic_data array
            new_episodic_data = new_episodic_data.at[
                active_trajectories_indexes,
                replay_buffer.timestep_positions,
            ].set(inserted_indexes)

            # Step 3: update the counters
            dones = flattened_transitions[
                :, (2 * (self.transition.observation_dim) + 1)
            ].ravel()

            # Increment the trajectory counter if done
            new_trajectory_positions = replay_buffer.trajectory_positions + dones

            # Within a trajectory, increment position if not done, else reset position
            new_timestep_positions = jnp.where(
                dones, jnp.zeros_like(dones), 1 + replay_buffer.timestep_positions
            )

            # Update the insertion position in the main buffer
            new_current_position = (
                replay_buffer.current_position + self.env_batch_size
            ) % self.buffer_size

            # Update the size counter of the main buffer
            new_current_size = jnp.minimum(
                replay_buffer.current_size + self.env_batch_size, self.buffer_size
            )

            # Update the size of the episodic data buffer
            new_current_episodic_data_size = jnp.minimum(
                jnp.min(replay_buffer.trajectory_positions + 1) * self.env_batch_size,
                self.num_trajectories,
            )

            replay_buffer = replay_buffer.replace(
                timestep_positions=jnp.array(new_timestep_positions, dtype=int),
                trajectory_positions=jnp.array(new_trajectory_positions, dtype=int),
                data=new_data,
                current_position=jnp.array(new_current_position, dtype=int),
                episodic_data=new_episodic_data,
                current_size=new_current_size,
                current_episodic_data_size=jnp.array(
                    new_current_episodic_data_size, dtype=int
                ),
            )
            return replay_buffer, None

        flattened_transitions = transitions.flatten()

        flattened_transitions = flattened_transitions.reshape(
            (-1, self.env_batch_size, flattened_transitions.shape[-1])
        )

        replay_buffer, _ = jax.lax.scan(
            insert_one_transition,
            self,
            flattened_transitions,
        )

        replay_buffer = replay_buffer.compute_returns()
        return replay_buffer  # type: ignore

    def compute_returns(
        self,
    ) -> TrajectoryBuffer:
        """Computes the return for each episode in the buffer.

        Returns:
            The buffer with the returns updated.
        """

        reward_idx = 2 * self.transition.observation_dim
        indexes = self.episodic_data
        rewards = self.data[:, reward_idx]
        episodic_returns = jnp.where(
            jnp.isnan(indexes),
            0,
            rewards[jnp.array(jnp.nan_to_num(indexes, 0), dtype=int)],
        ).sum(axis=1)

        values = episodic_returns[:, None].repeat(self.episode_length, axis=1)
        returns = self.returns.at[
            jnp.array(jnp.nan_to_num(indexes, nan=-1), dtype=int)
        ].set(values)
        returns = returns.at[-1].set(jnp.nan)
        return self.replace(returns=returns)  # type: ignore
