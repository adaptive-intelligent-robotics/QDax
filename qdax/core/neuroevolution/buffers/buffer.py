from __future__ import annotations

from functools import partial
from typing import Tuple

import flax
import jax
import jax.numpy as jnp

from qdax.types import Action, Done, Observation, Reward, RNGKey, StateDescriptor


class Transition(flax.struct.PyTreeNode):
    """Stores data corresponding to a transition collected by a classic RL algorithm."""

    obs: Observation
    next_obs: Observation
    rewards: Reward
    dones: Done
    truncations: jnp.ndarray  # Indicates if an episode has reached max time step
    actions: Action

    @property
    def observation_dim(self) -> int:
        """
        Returns:
            the dimension of the observation
        """
        return self.obs.shape[-1]  # type: ignore

    @property
    def action_dim(self) -> int:
        """
        Returns:
            the dimension of the action
        """
        return self.actions.shape[-1]  # type: ignore

    @property
    def flatten_dim(self) -> int:
        """
        Returns:
            the dimension of the transition once flattened.

        """
        flatten_dim = 2 * self.observation_dim + self.action_dim + 3
        return flatten_dim

    def flatten(self) -> jnp.ndarray:
        """
        Returns:
            a jnp.ndarray that corresponds to the flattened transition.
        """
        flatten_transition = jnp.concatenate(
            [
                self.obs,
                self.next_obs,
                jnp.expand_dims(self.rewards, axis=-1),
                jnp.expand_dims(self.dones, axis=-1),
                jnp.expand_dims(self.truncations, axis=-1),
                self.actions,
            ],
            axis=-1,
        )
        return flatten_transition

    @classmethod
    def from_flatten(
        cls,
        flattened_transition: jnp.ndarray,
        transition: Transition,
    ) -> Transition:
        """
        Creates a transition from a flattened transition in a jnp.ndarray.

        Args:
            flattened_transition: flattened transition in a jnp.ndarray of shape
                (batch_size, flatten_dim)
            transition: a transition object (might be a dummy one) to
                get the dimensions right

        Returns:
            a Transition object
        """
        obs_dim = transition.observation_dim
        action_dim = transition.action_dim
        obs = flattened_transition[:, :obs_dim]
        next_obs = flattened_transition[:, obs_dim : (2 * obs_dim)]
        rewards = jnp.ravel(flattened_transition[:, (2 * obs_dim) : (2 * obs_dim + 1)])
        dones = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 1) : (2 * obs_dim + 2)]
        )
        truncations = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 2) : (2 * obs_dim + 3)]
        )
        actions = flattened_transition[
            :, (2 * obs_dim + 3) : (2 * obs_dim + 3 + action_dim)
        ]
        return cls(
            obs=obs,
            next_obs=next_obs,
            rewards=rewards,
            dones=dones,
            truncations=truncations,
            actions=actions,
        )

    @classmethod
    def init_dummy(cls, observation_dim: int, action_dim: int) -> Transition:
        """
        Initialize a dummy transition that then can be passed to constructors to get
        all shapes right.

        Args:
            observation_dim: observation dimension
            action_dim: action dimension

        Returns:
            a dummy transition
        """
        dummy_transition = Transition(
            obs=jnp.zeros(shape=(1, observation_dim)),
            next_obs=jnp.zeros(shape=(1, observation_dim)),
            rewards=jnp.zeros(shape=(1,)),
            dones=jnp.zeros(shape=(1,)),
            truncations=jnp.zeros(shape=(1,)),
            actions=jnp.zeros(shape=(1, action_dim)),
        )
        return dummy_transition


class QDTransition(Transition):
    """Stores data corresponding to a transition collected by a QD algorithm."""

    state_desc: StateDescriptor
    next_state_desc: StateDescriptor

    @property
    def state_descriptor_dim(self) -> int:
        """
        Returns:
            the dimension of the state descriptors.

        """
        return self.state_desc.shape[-1]  # type: ignore

    @property
    def flatten_dim(self) -> int:
        """
        Returns:
            the dimension of the transition once flattened.

        """
        flatten_dim = (
            2 * self.observation_dim
            + self.action_dim
            + 3
            + 2 * self.state_descriptor_dim
        )
        return flatten_dim

    def flatten(self) -> jnp.ndarray:
        """
        Returns:
            a jnp.ndarray that corresponds to the flattened transition.
        """
        flatten_transition = jnp.concatenate(
            [
                self.obs,
                self.next_obs,
                jnp.expand_dims(self.rewards, axis=-1),
                jnp.expand_dims(self.dones, axis=-1),
                jnp.expand_dims(self.truncations, axis=-1),
                self.actions,
                self.state_desc,
                self.next_state_desc,
            ],
            axis=-1,
        )
        return flatten_transition

    @classmethod
    def from_flatten(
        cls,
        flattened_transition: jnp.ndarray,
        transition: QDTransition,
    ) -> QDTransition:
        """
        Creates a transition from a flattened transition in a jnp.ndarray.

        Args:
            flattened_transition: flattened transition in a jnp.ndarray of shape
                (batch_size, flatten_dim)
            transition: a transition object (might be a dummy one) to
                get the dimensions right

        Returns:
            a Transition object
        """
        obs_dim = transition.observation_dim
        action_dim = transition.action_dim
        desc_dim = transition.state_descriptor_dim

        obs = flattened_transition[:, :obs_dim]
        next_obs = flattened_transition[:, obs_dim : (2 * obs_dim)]
        rewards = jnp.ravel(flattened_transition[:, (2 * obs_dim) : (2 * obs_dim + 1)])
        dones = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 1) : (2 * obs_dim + 2)]
        )
        truncations = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 2) : (2 * obs_dim + 3)]
        )
        actions = flattened_transition[
            :, (2 * obs_dim + 3) : (2 * obs_dim + 3 + action_dim)
        ]
        state_desc = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim) : (2 * obs_dim + 3 + action_dim + desc_dim),
        ]
        next_state_desc = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim + desc_dim) : (
                2 * obs_dim + 3 + action_dim + 2 * desc_dim
            ),
        ]
        return cls(
            obs=obs,
            next_obs=next_obs,
            rewards=rewards,
            dones=dones,
            truncations=truncations,
            actions=actions,
            state_desc=state_desc,
            next_state_desc=next_state_desc,
        )

    @classmethod
    def init_dummy(  # type: ignore
        cls, observation_dim: int, action_dim: int, descriptor_dim: int
    ) -> QDTransition:
        """
        Initialize a dummy transition that then can be passed to constructors to get
        all shapes right.

        Args:
            observation_dim: observation dimension
            action_dim: action dimension

        Returns:
            a dummy transition
        """
        dummy_transition = QDTransition(
            obs=jnp.zeros(shape=(1, observation_dim)),
            next_obs=jnp.zeros(shape=(1, observation_dim)),
            rewards=jnp.zeros(shape=(1,)),
            dones=jnp.zeros(shape=(1,)),
            truncations=jnp.zeros(shape=(1,)),
            actions=jnp.zeros(shape=(1, action_dim)),
            state_desc=jnp.zeros(shape=(1, descriptor_dim)),
            next_state_desc=jnp.zeros(shape=(1, descriptor_dim)),
        )
        return dummy_transition


class ReplayBuffer(flax.struct.PyTreeNode):
    """
    A replay buffer where transitions are flattened before being stored.
    Transitions are unflatenned on the fly when sampled in the buffer.
    data shape: (buffer_size, transition_concat_shape)
    """

    data: jnp.ndarray
    buffer_size: int = flax.struct.field(pytree_node=False)
    transition: Transition

    current_position: jnp.ndarray = flax.struct.field()
    current_size: jnp.ndarray = flax.struct.field()

    @classmethod
    def init(
        cls,
        buffer_size: int,
        transition: Transition,
    ) -> ReplayBuffer:
        """
        The constructor of the buffer.

        Note: We have to define a classmethod instead of just doing it in post_init
        because post_init is called every time the dataclass is tree_mapped. This is a
        workaround proposed in https://github.com/google/flax/issues/1628.

        Args:
            buffer_size: the size of the replay buffer, e.g. 1e6
            transition: a transition object (might be a dummy one) to get
                the dimensions right
        """
        flatten_dim = transition.flatten_dim
        data = jnp.ones((buffer_size, flatten_dim)) * jnp.nan
        current_size = jnp.array(0, dtype=int)
        current_position = jnp.array(0, dtype=int)
        return cls(
            data=data,
            current_size=current_size,
            current_position=current_position,
            buffer_size=buffer_size,
            transition=transition,
        )

    @partial(jax.jit, static_argnames=("sample_size",))
    def sample(
        self,
        random_key: RNGKey,
        sample_size: int,
    ) -> Tuple[Transition, RNGKey]:
        """
        Sample a batch of transitions in the replay buffer.
        """
        random_key, subkey = jax.random.split(random_key)
        idx = jax.random.randint(
            subkey,
            shape=(sample_size,),
            minval=0,
            maxval=self.current_size,
        )
        samples = jnp.take(self.data, idx, axis=0, mode="clip")
        transitions = self.transition.__class__.from_flatten(samples, self.transition)
        return transitions, random_key

    @jax.jit
    def insert(self, transitions: Transition) -> ReplayBuffer:
        """
        Insert a batch of transitions in the replay buffer. The transitions are
        flattened before insertion.

        Args:
            transitions: A transition object in which each field is assumed to have
                a shape (batch_size, field_dim).
        """
        flattened_transitions = transitions.flatten()
        flattened_transitions = flattened_transitions.reshape(
            (-1, flattened_transitions.shape[-1])
        )
        num_transitions = flattened_transitions.shape[0]
        max_replay_size = self.buffer_size

        # Make sure update is not larger than the maximum replay size.
        if num_transitions > max_replay_size:
            raise ValueError(
                "Trying to insert a batch of samples larger than the maximum replay "
                f"size. num_samples: {num_transitions}, "
                f"max replay size {max_replay_size}"
            )

        # get current position
        position = self.current_position

        # check if there is an overlap
        roll = jnp.minimum(0, max_replay_size - position - num_transitions)

        # roll the data to avoid overlap
        data = jnp.roll(self.data, roll, axis=0)

        # update the position accordingly
        new_position = position + roll

        # replace old data by the new one
        new_data = jax.lax.dynamic_update_slice_in_dim(
            data,
            flattened_transitions,
            start_index=new_position,
            axis=0,
        )

        # update the position and the size
        new_position = (new_position + num_transitions) % max_replay_size
        new_size = jnp.minimum(self.current_size + num_transitions, max_replay_size)

        # update the replay buffer
        replay_buffer = self.replace(
            current_position=new_position,
            current_size=new_size,
            data=new_data,
        )

        return replay_buffer  # type: ignore
