from functools import partial

import flax
import jax
import jax.numpy as jnp
from flax import struct

from qdax.types import RNGKey
from qdax.utils.mdp_utils import Transition


@flax.struct.dataclass
class FlatBuffer:
    """
    A replay buffer where transitions are flattened before being stored.
    Transitions are unflatenned on the fly when sampled in the buffer.
    data shape: (buffer_size, transition_concat_shape)
    """

    data: jnp.ndarray
    buffer_size: int = struct.field(pytree_node=False)
    transition: Transition

    current_position: jnp.ndarray = struct.field()
    current_size: jnp.ndarray = struct.field()

    @classmethod
    def init(
        cls,
        buffer_size: int,
        transition: Transition,
    ):
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
    ) -> Transition:
        """
        Sample a batch of transitions in the replay buffer.
        """
        idx = jax.random.randint(
            random_key,
            shape=(sample_size,),
            minval=0,
            maxval=self.current_size,
        )
        samples = jnp.take(self.data, idx, axis=0, mode="clip")
        transitions = self.transition.__class__.from_flatten(samples, self.transition)
        return transitions

    @jax.jit
    def insert(self, transitions: Transition) -> "FlatBuffer":
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

        new_current_position = self.current_position + num_transitions
        new_current_size = jnp.minimum(
            self.current_size + num_transitions, max_replay_size
        )
        new_data = jax.lax.dynamic_update_slice_in_dim(
            self.data,
            flattened_transitions,
            start_index=self.current_position % max_replay_size,
            axis=0,
        )

        replay_buffer = self.replace(
            current_position=new_current_position,
            current_size=new_current_size,
            data=new_data,
        )

        return replay_buffer
