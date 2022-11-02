import jax.numpy as jnp
import pytest

from qdax.core.containers.archive import Archive, score_euclidean_novelty

a = jnp.array([[0.0, 0.0], [0.0, 1.0]])
b = jnp.array([[1.0, 0.0], [1.0, 1.0]])
c = jnp.array([[2.0, 0.0], [2.0, 1.0]])
d = jnp.array([[0.0, 2.0], [1.0, 2.0]])
e = jnp.array([[2.0, 2.0], [3.0, 2.0]])
f = jnp.array([[0.1, 0.0], [0.9, 0.0]])

# TODO: missing cases


def test_archive() -> None:
    """Test basic functions of the archive in simple
    cases. Small threshold and enough size to avoid
    replacement in the archive.
    """
    fake_border = jnp.inf
    # create archive
    archive = Archive.create(0.1, 2, 20, fake_border)
    expected_init_data = (
        jnp.ones((archive.max_size, archive.state_descriptor_size)) * fake_border
    )

    # check initial data
    pytest.assume(archive.current_position == 0)
    pytest.assume((archive.data == expected_init_data).all())

    # check single insertion
    new_elem = jnp.array([10.0, 10.0])
    expected_new_data = expected_init_data.at[0].set(new_elem)
    print("Expected new data : ", expected_new_data)

    archive = archive._single_insertion(new_elem)
    print("Archive data : ", archive.data)
    pytest.assume(archive.current_position == 1)
    pytest.assume(jnp.allclose(archive.data, expected_new_data, atol=1e-6))

    # check conditioned single insertion
    new_elem = jnp.array([20.0, 20.0])

    # when False
    archive, _ = archive._conditioned_single_insertion(False, new_elem)
    pytest.assume(archive.current_position == 1)
    pytest.assume(jnp.allclose(archive.data, expected_new_data, atol=1e-6))

    # when True
    expected_new_data = expected_new_data.at[1].set(new_elem)
    archive, _ = archive._conditioned_single_insertion(True, new_elem)
    pytest.assume(archive.current_position == 2)
    pytest.assume(jnp.allclose(archive.data, expected_new_data, atol=1e-6))

    # check insertion in archive
    import jax

    with jax.disable_jit():
        expected_new_data = expected_new_data.at[2:4].set(a)
        archive = archive.insert(a)

    print("Archive: ", archive)
    print("Expected data: ", expected_new_data)

    pytest.assume(archive.current_position == 4)
    pytest.assume(jnp.allclose(archive.data, expected_new_data, atol=1e-6))

    # check new insertion in archive
    expected_new_data = expected_new_data.at[4:6].set(b)
    archive = archive.insert(b)
    pytest.assume(archive.current_position == 6)
    pytest.assume(jnp.allclose(archive.data, expected_new_data, atol=1e-6))


def test_archive_overflow() -> None:
    """Test archive insertion when the maximal size
    of the archive has been reached."""

    fake_border = jnp.inf

    # create archive
    archive = Archive.create(0.1, 2, 6, fake_border)

    # add elements till the limit
    archive = archive.insert(a).insert(b).insert(c)
    expected_data = jnp.concatenate([a, b, c], axis=0)
    pytest.assume(archive.current_position == 6)
    pytest.assume(jnp.allclose(archive.data, expected_data, atol=1e-6))

    # add another element to the archive
    archive = archive.insert(d)
    expected_data = expected_data.at[0:2].set(d)
    pytest.assume(archive.current_position == 8)
    pytest.assume(jnp.allclose(archive.data, expected_data, atol=1e-6))

    # add yet another element to the archive
    archive = archive.insert(e)
    expected_data = expected_data.at[2:4].set(e)
    pytest.assume(archive.current_position == 10)
    pytest.assume(jnp.allclose(archive.data, expected_data, atol=1e-6))


def test_archive_high_threshold() -> None:
    """Test the archive insertion mechanism in the case where
    the acceptance threshold prevent some elements from being inserted"""
    fake_border = jnp.inf
    acceptance_threshold = jnp.sqrt(0.5)
    # create archive
    archive = Archive.create(acceptance_threshold, 2, 6, fake_border)

    # add some elements
    archive = archive.insert(a)
    pytest.assume(archive.current_position == 2)
    old_archive_data = archive.data

    # create similar elements to a
    a_p = a + (acceptance_threshold / 10)

    # try to insert it to the archive
    archive = archive.insert(a_p)

    # check that nothing was added
    pytest.assume(archive.current_position == 2)
    pytest.assume(jnp.allclose(archive.data, old_archive_data, atol=1e-6))


def test_archive_filter_at_entry() -> None:
    """Test that the filtering of near points works at the entry as well"""
    fake_border = jnp.inf
    archive = Archive.create(jnp.sqrt(1.5), 2, 10, fake_border)

    a = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.5], [1.0, 1.0]])
    archive = archive.insert(a)

    expected_data = (
        jnp.ones((archive.max_size, archive.state_descriptor_size)) * fake_border
    )
    expected_data = expected_data.at[0:2].set(jnp.array([[1.0, 0.0], [0.0, 1.0]]))
    pytest.assume(archive.current_position == 2)
    pytest.assume(jnp.allclose(archive.data, expected_data, atol=1e-6))


def test_euclidean_scorer() -> None:
    # create the novelty scorer
    fake_border = jnp.inf
    num_nearest_neighb = 2
    scaling_ratio = 1

    # create archive
    archive = Archive.create(0.1, 2, 10, fake_border)

    # insert a in the archive
    archive = archive.insert(a)

    # create an element
    ab = jnp.array([[0.0, 0.5]])

    # score ab
    ab_rewards = score_euclidean_novelty(archive, ab, num_nearest_neighb, scaling_ratio)
    pytest.assume(jnp.allclose(ab_rewards, jnp.array([0.25]), atol=1e-6))

    # try to score two elements at the same time
    b_rewards = score_euclidean_novelty(archive, b, num_nearest_neighb, scaling_ratio)
    expected_rewards = jnp.array([1.5, 1.5])
    pytest.assume(jnp.allclose(b_rewards, expected_rewards, atol=1e-6))
