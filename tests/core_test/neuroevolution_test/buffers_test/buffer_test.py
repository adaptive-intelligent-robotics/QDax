import jax
import jax.numpy as jnp
import pytest

from qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer


def test_insert() -> None:
    """Tests if an insert of a dummy transition results in a buffer of size 1"""
    observation_size = 2
    action_size = 8
    buffer_size = 10

    # Initialize buffer
    dummy_transition = QDTransition.init_dummy(
        observation_dim=observation_size,
        action_dim=action_size,
        descriptor_dim=0,
    )
    replay_buffer = ReplayBuffer.init(
        buffer_size=buffer_size, transition=dummy_transition
    )

    replay_buffer = replay_buffer.insert(dummy_transition)
    pytest.assume(replay_buffer.current_size == 1)


def test_insert_batch() -> None:
    """Tests if inserting transitions such that we exceed the max size
    of the buffer leads to the desired behavior."""
    observation_size = 2
    action_size = 8
    buffer_size = 5

    # Initialize buffer
    dummy_transition = QDTransition.init_dummy(
        observation_dim=observation_size,
        action_dim=action_size,
        descriptor_dim=0,
    )
    replay_buffer = ReplayBuffer.init(
        buffer_size=buffer_size, transition=dummy_transition
    )

    simple_transition = jax.tree_util.tree_map(
        lambda x: x.repeat(3, axis=0), dummy_transition
    )
    simple_transition = simple_transition.replace(rewards=jnp.arange(3))
    data = QDTransition.from_flatten(replay_buffer.data, dummy_transition)
    pytest.assume(
        jnp.array_equal(data.rewards, jnp.array([jnp.nan] * 5), equal_nan=True).all()
    )

    replay_buffer = replay_buffer.insert(simple_transition)
    data = QDTransition.from_flatten(replay_buffer.data, dummy_transition)
    pytest.assume(
        jnp.array_equal(
            data.rewards, jnp.array([0, 1, 2, jnp.nan, jnp.nan]), equal_nan=True
        ).all()
    )

    simple_transition_2 = simple_transition.replace(rewards=jnp.arange(3, 6))
    replay_buffer = replay_buffer.insert(simple_transition_2)
    data = QDTransition.from_flatten(replay_buffer.data, dummy_transition)
    pytest.assume(
        jnp.array_equal(data.rewards, jnp.array([1, 2, 3, 4, 5]), equal_nan=True).all()
    )


def test_sample() -> None:
    """
    Tests if sampled transitions have valid shape.
    """
    observation_size = 2
    action_size = 8
    buffer_size = 5

    # Initialize buffer
    dummy_transition = QDTransition.init_dummy(
        observation_dim=observation_size,
        action_dim=action_size,
        descriptor_dim=0,
    )
    replay_buffer = ReplayBuffer.init(
        buffer_size=buffer_size, transition=dummy_transition
    )

    simple_transition = jax.tree_util.tree_map(
        lambda x: x.repeat(3, axis=0), dummy_transition
    )
    simple_transition = simple_transition.replace(rewards=jnp.arange(3))

    replay_buffer = replay_buffer.insert(simple_transition)
    random_key = jax.random.PRNGKey(0)

    samples, random_key = replay_buffer.sample(random_key, 3)

    samples_shapes = jax.tree_util.tree_map(lambda x: x.shape, samples)
    transition_shapes = jax.tree_util.tree_map(lambda x: x.shape, simple_transition)
    pytest.assume((samples_shapes == transition_shapes))
