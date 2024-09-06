import jax.numpy as jnp
import pytest

from qdax.core.neuroevolution.buffers.buffer import QDTransition, Transition
from qdax.core.neuroevolution.buffers.trajectory_buffer import TrajectoryBuffer


def update_returns_naive(buffer: TrajectoryBuffer) -> jnp.ndarray:
    """Loops over the episodic_data of the buffer to compute the per_episode_reward and
    to assign to each transition its per_episode_reward. It is tested to compute the
    same returns vector than trajectory_buffer.compute_returns().

    Args:
        buffer: a trajectory buffer

    Returns:
        an array of size (buffer_size,) that contains the episodic return for each
        transition
    """
    episodic_data = buffer.episodic_data
    num_trajs, episode_length = episodic_data.shape

    per_episode_return = jnp.ones(num_trajs) * -jnp.inf

    for i in range(num_trajs):
        episode_idx = buffer.episodic_data[i]
        episode_idx = episode_idx[~jnp.isnan(episode_idx)]
        if len(episode_idx) == 0:
            episode_return = -jnp.inf
        else:
            episode_return = jnp.sum(
                buffer.data[:, 0][jnp.array(episode_idx, dtype=int)]
            )
        per_episode_return = per_episode_return.at[i].set(episode_return)

    returns = buffer.returns

    for i in range(num_trajs):
        for j in range(episode_length):
            k = episodic_data[i, j]
            if not jnp.isnan(k):
                returns = returns.at[int(k)].set(per_episode_return[i])
    returns = returns.at[-1].set(jnp.nan)
    return returns


def test_trajectory_compute_returns() -> None:
    """Tests if the TrajectoryBuffer reward computation is consistent with a naive
    (and slow) summation of the rewards.
    """
    buffer_size = 12
    env_batch_size = 1
    episode_length = 3

    transition = QDTransition.init_dummy(
        observation_dim=0, action_dim=0, descriptor_dim=0
    )

    traj_buffer = TrajectoryBuffer.init(
        buffer_size=buffer_size,
        env_batch_size=env_batch_size,
        episode_length=episode_length,
        transition=transition,
    )

    for i in range(13):
        if i == 2 or i == 4 or i == 7 or i == 10:
            dones = jnp.ones(shape=(env_batch_size,))
            rewards = jnp.ones(shape=(env_batch_size,)) * 5
        else:
            dones = jnp.zeros(shape=(env_batch_size,))
            rewards = jnp.ones(shape=(env_batch_size,))
        transition = QDTransition(
            obs=jnp.zeros(shape=(env_batch_size, 0)),
            next_obs=jnp.zeros(shape=(env_batch_size, 0)),
            rewards=rewards,
            dones=dones,
            truncations=jnp.zeros(shape=(env_batch_size,)),
            actions=jnp.zeros(shape=(env_batch_size, 0)),
            state_desc=jnp.zeros(shape=(env_batch_size, 0)),
            next_state_desc=jnp.zeros(shape=(env_batch_size, 0)),
        )
        traj_buffer = traj_buffer.insert(transition)
        naive_returns = update_returns_naive(traj_buffer)
        traj_buffer = traj_buffer.compute_returns()
        print(naive_returns)
        print(traj_buffer.returns)
        pytest.assume(
            jnp.array_equal(naive_returns, traj_buffer.returns, equal_nan=True).all(),
            "Returns computed by naive loop are not consistent with returns \
             computed with fancy indexing",
        )

    assert jnp.array_equal(
        traj_buffer.returns,
        jnp.array([2, 7, 7, 6, 6, 7, 7, 7, 7, 7, 7, 2, jnp.nan]),
        equal_nan=True,
    )


def test_trajectory_buffer_insert() -> None:
    """
    This test is a simulation where
    multiple transitions are inserted into the buffer, it checks
    if anything is incoherent.
    """
    observation_size = 2
    action_size = 1
    env_batch_size = 3
    buffer_size = 9
    episode_length = 3

    # Multi step insert

    transition = Transition.init_dummy(
        observation_dim=observation_size, action_dim=action_size
    )

    buffer = TrajectoryBuffer.init(
        buffer_size=buffer_size,
        env_batch_size=env_batch_size,
        episode_length=episode_length,
        transition=transition,
    )

    obs = jnp.ones((4 * env_batch_size, observation_size))
    dones = jnp.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0])
    actions = jnp.zeros((4 * env_batch_size, action_size))
    rewards = jnp.ones((4 * env_batch_size,))
    transitions = Transition(
        obs=obs,
        next_obs=obs,
        actions=actions,
        dones=dones,
        rewards=rewards,
        truncations=dones,
    )

    buffer = buffer.insert(transitions)

    dones = jnp.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    transitions = Transition(
        obs=obs,
        next_obs=obs,
        actions=actions,
        dones=dones,
        rewards=rewards,
        truncations=dones,
    )

    buffer = buffer.insert(transitions)
    multy_step_episodic_data = buffer.episodic_data

    # Single step insert

    obs = obs.reshape((-1, env_batch_size, observation_size))
    dones = dones.reshape((-1, env_batch_size))
    actions = actions.reshape((-1, env_batch_size, action_size))
    rewards = rewards.reshape((-1, env_batch_size))

    transition = Transition.init_dummy(
        observation_dim=observation_size, action_dim=action_size
    )

    buffer = TrajectoryBuffer.init(
        buffer_size=buffer_size,
        env_batch_size=env_batch_size,
        episode_length=episode_length,
        transition=transition,
    )

    for i in range(4):
        transitions = Transition(
            obs=obs[i],
            next_obs=obs[i],
            actions=actions[i],
            dones=dones[i],
            rewards=rewards[i],
            truncations=dones[i],
        )
        buffer = buffer.insert(transitions)

    dones = jnp.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    dones = dones.reshape((-1, env_batch_size))
    for i in range(4):
        transitions = Transition(
            obs=obs[i],
            next_obs=obs[i],
            actions=actions[i],
            dones=dones[i],
            rewards=rewards[i],
            truncations=dones[i],
        )

        buffer = buffer.insert(transitions)

    print(buffer.episodic_data)

    pytest.assume(
        jnp.array_equal(
            buffer.episodic_data,
            multy_step_episodic_data,
            equal_nan=True,
        ),
        "Episodic data when transitions are added sequentially is not consistent to \
        when they are added as batch.",
    )

    pytest.assume(
        jnp.array_equal(
            buffer.episodic_data,
            jnp.array(
                [[6.0, 0.0, 3.0], [7.0, 1.0, 4.0], [8.0, 2.0, 5.0]]
            ),  # Handchecked
            equal_nan=True,
        ),
        "Expected a different output",
    )
