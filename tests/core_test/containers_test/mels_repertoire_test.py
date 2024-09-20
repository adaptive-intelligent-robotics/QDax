import jax.numpy as jnp
import pytest

from qdax.core.containers.mels_repertoire import MELSRepertoire
from qdax.custom_types import ExtraScores


def test_add_to_mels_repertoire() -> None:
    """Test several additions to the MELSRepertoire, including adding a solution
    and overwriting it by adding multiple solutions."""
    genotype_size = 12
    num_centroids = 4
    num_descriptors = 2

    # create a repertoire instance
    repertoire = MELSRepertoire(
        genotypes=jnp.zeros(shape=(num_centroids, genotype_size)),
        fitnesses=jnp.ones(shape=(num_centroids,)) * (-jnp.inf),
        descriptors=jnp.zeros(shape=(num_centroids, num_descriptors)),
        centroids=jnp.array(
            [
                [1.0, 1.0],
                [2.0, 1.0],
                [2.0, 2.0],
                [1.0, 2.0],
            ]
        ),
        spreads=jnp.full(shape=(num_centroids,), fill_value=jnp.inf),
    )

    #
    # Test 1: Insert a single solution.
    #

    # create fake genotypes and scores to add
    fake_genotypes = jnp.ones(shape=(1, genotype_size))
    # each solution gets two fitnesses and two descriptors
    fake_fitnesses = jnp.array([[0.0, 0.0]])
    fake_descriptors = jnp.array([[[0.0, 1.0], [1.0, 1.0]]])
    fake_extra_scores: ExtraScores = {}

    # do an addition
    repertoire = repertoire.add(
        fake_genotypes, fake_descriptors, fake_fitnesses, fake_extra_scores
    )

    # check that the repertoire looks as expected
    expected_genotypes = jnp.array(
        [
            [1.0 for _ in range(genotype_size)],
            [0.0 for _ in range(genotype_size)],
            [0.0 for _ in range(genotype_size)],
            [0.0 for _ in range(genotype_size)],
        ]
    )
    expected_fitnesses = jnp.array([0.0, -jnp.inf, -jnp.inf, -jnp.inf])
    expected_descriptors = jnp.array(
        [
            [1.0, 1.0],  # Centroid coordinates.
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    expected_spreads = jnp.array([1.0, jnp.inf, jnp.inf, jnp.inf])

    # check values
    pytest.assume(jnp.allclose(repertoire.genotypes, expected_genotypes, atol=1e-6))
    pytest.assume(jnp.allclose(repertoire.fitnesses, expected_fitnesses, atol=1e-6))
    pytest.assume(jnp.allclose(repertoire.descriptors, expected_descriptors, atol=1e-6))
    pytest.assume(jnp.allclose(repertoire.spreads, expected_spreads, atol=1e-6))

    #
    # Test 2: Adding solutions into the same cell as above.
    #

    # create fake genotypes and scores to add
    fake_genotypes = jnp.concatenate(
        (
            jnp.full(shape=(1, genotype_size), fill_value=2.0),
            jnp.full(shape=(1, genotype_size), fill_value=3.0),
        ),
        axis=0,
    )
    # Each solution gets two fitnesses and two descriptors (i.e. num_evals = 2). One
    # solution has fitness 1.0 and spread 0.75, while the other has fitness 0.5 and
    # spread 0.5. Thus, neither solution dominates the other (by having both higher
    # fitness and lower spread). However, both solutions would be valid candidates for
    # the archive due to dominating the current solution there.
    fake_fitnesses = jnp.array([[1.0, 1.0], [0.5, 0.5]])
    fake_descriptors = jnp.array([[[1.0, 0.25], [1.0, 1.0]], [[1.0, 0.5], [1.0, 1.0]]])
    fake_extra_scores: ExtraScores = {}

    # do an addition
    repertoire = repertoire.add(
        fake_genotypes, fake_descriptors, fake_fitnesses, fake_extra_scores
    )

    # Either solution may be added due to the behavior of jax.at[].set():
    # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
    # Thus, we provide possible values for each scenario.

    # check that the repertoire looks like expected
    expected_genotypes_1 = jnp.array(
        [
            [2.0 for _ in range(genotype_size)],
            [0.0 for _ in range(genotype_size)],
            [0.0 for _ in range(genotype_size)],
            [0.0 for _ in range(genotype_size)],
        ]
    )
    expected_fitnesses_1 = jnp.array([1.0, -jnp.inf, -jnp.inf, -jnp.inf])
    expected_descriptors_1 = jnp.array(
        [
            [1.0, 1.0],  # Centroid coordinates.
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    expected_spreads_1 = jnp.array([0.75, jnp.inf, jnp.inf, jnp.inf])

    expected_genotypes_2 = jnp.array(
        [
            [3.0 for _ in range(genotype_size)],
            [0.0 for _ in range(genotype_size)],
            [0.0 for _ in range(genotype_size)],
            [0.0 for _ in range(genotype_size)],
        ]
    )
    expected_fitnesses_2 = jnp.array([0.5, -jnp.inf, -jnp.inf, -jnp.inf])
    expected_descriptors_2 = jnp.array(
        [
            [1.0, 1.0],  # Centroid coordinates.
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    expected_spreads_2 = jnp.array([0.5, jnp.inf, jnp.inf, jnp.inf])

    # check values
    pytest.assume(
        jnp.allclose(repertoire.genotypes, expected_genotypes_1, atol=1e-6)
        or jnp.allclose(repertoire.genotypes, expected_genotypes_2, atol=1e-6)
    )

    if jnp.allclose(repertoire.genotypes, expected_genotypes_1, atol=1e-6):
        pytest.assume(
            jnp.allclose(repertoire.genotypes, expected_genotypes_1, atol=1e-6)
        )
        pytest.assume(
            jnp.allclose(repertoire.fitnesses, expected_fitnesses_1, atol=1e-6)
        )
        pytest.assume(
            jnp.allclose(repertoire.descriptors, expected_descriptors_1, atol=1e-6)
        )
        pytest.assume(jnp.allclose(repertoire.spreads, expected_spreads_1, atol=1e-6))
    elif jnp.allclose(repertoire.genotypes, expected_genotypes_2, atol=1e-6):
        pytest.assume(
            jnp.allclose(repertoire.genotypes, expected_genotypes_2, atol=1e-6)
        )
        pytest.assume(
            jnp.allclose(repertoire.fitnesses, expected_fitnesses_2, atol=1e-6)
        )
        pytest.assume(
            jnp.allclose(repertoire.descriptors, expected_descriptors_2, atol=1e-6)
        )
        pytest.assume(jnp.allclose(repertoire.spreads, expected_spreads_2, atol=1e-6))


def test_add_with_single_eval() -> None:
    """Tries adding with a single evaluation.

    This is a special case because the spread defaults to 0.
    """
    genotype_size = 12
    num_centroids = 4
    num_descriptors = 2

    # create a repertoire instance
    repertoire = MELSRepertoire(
        genotypes=jnp.zeros(shape=(num_centroids, genotype_size)),
        fitnesses=jnp.ones(shape=(num_centroids,)) * (-jnp.inf),
        descriptors=jnp.zeros(shape=(num_centroids, num_descriptors)),
        centroids=jnp.array(
            [
                [1.0, 1.0],
                [2.0, 1.0],
                [2.0, 2.0],
                [1.0, 2.0],
            ]
        ),
        spreads=jnp.full(shape=(num_centroids,), fill_value=jnp.inf),
    )

    # Insert a single solution with only one eval.

    # create fake genotypes and scores to add
    fake_genotypes = jnp.ones(shape=(1, genotype_size))
    # the solution gets one fitness and one descriptor.
    fake_fitnesses = jnp.array([[0.0]])
    fake_descriptors = jnp.array([[[0.0, 1.0]]])
    fake_extra_scores: ExtraScores = {}

    # do an addition
    repertoire = repertoire.add(
        fake_genotypes, fake_descriptors, fake_fitnesses, fake_extra_scores
    )

    # check that the repertoire looks as expected
    expected_genotypes = jnp.array(
        [
            [1.0 for _ in range(genotype_size)],
            [0.0 for _ in range(genotype_size)],
            [0.0 for _ in range(genotype_size)],
            [0.0 for _ in range(genotype_size)],
        ]
    )
    expected_fitnesses = jnp.array([0.0, -jnp.inf, -jnp.inf, -jnp.inf])
    expected_descriptors = jnp.array(
        [
            [1.0, 1.0],  # Centroid coordinates.
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    # Spread should be 0 since there's only one eval.
    expected_spreads = jnp.array([0.0, jnp.inf, jnp.inf, jnp.inf])

    # check values
    pytest.assume(jnp.allclose(repertoire.genotypes, expected_genotypes, atol=1e-6))
    pytest.assume(jnp.allclose(repertoire.fitnesses, expected_fitnesses, atol=1e-6))
    pytest.assume(jnp.allclose(repertoire.descriptors, expected_descriptors, atol=1e-6))
    pytest.assume(jnp.allclose(repertoire.spreads, expected_spreads, atol=1e-6))
