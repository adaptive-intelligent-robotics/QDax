import jax
import jax.numpy as jnp

from qdax.core.containers.unstructured_repertoire import UnstructuredRepertoire


def _stored(repertoire: UnstructuredRepertoire) -> jnp.ndarray:
    """Returns a mask of the slots of the repertoire that hold a solution."""
    return jnp.ravel(repertoire.fitnesses) > -jnp.inf


def test_better_neighbour_replaces_solution() -> None:
    """A solution within l_value of a single stored one, with a higher fitness,
    takes its slot instead of being added next to it."""
    repertoire = UnstructuredRepertoire.init(
        genotypes=jnp.array([[1.0], [2.0]]),
        fitnesses=jnp.array([0.0, 0.0]),
        descriptors=jnp.array([[0.5, 0.5], [0.9, 0.9]]),
        l_value=jnp.array([0.15]),
        max_size=10,
    )

    # The first solution is 0.05 from the stored solution 1 and better; the
    # second one is far from every stored solution.
    repertoire = repertoire.add(
        batch_of_genotypes=jnp.array([[3.0], [4.0]]),
        batch_of_descriptors=jnp.array([[0.55, 0.5], [0.1, 0.1]]),
        batch_of_fitnesses=jnp.array([1.0, 0.0]),
    )

    stored_genotypes = jnp.sort(repertoire.genotypes[_stored(repertoire), 0])
    assert jnp.array_equal(stored_genotypes, jnp.array([2.0, 3.0, 4.0]))


def test_stored_solutions_are_l_value_apart() -> None:
    """After batches of random solutions, no two stored descriptors are closer
    than l_value."""
    l_value = 0.15
    batch_size = 10
    num_batches = 15
    key_fitnesses, key_descriptors = jax.random.split(jax.random.key(0))
    fitnesses = jax.random.normal(key_fitnesses, (num_batches, batch_size))
    descriptors = jax.random.uniform(key_descriptors, (num_batches, batch_size, 2))
    genotypes = jnp.arange(num_batches * batch_size, dtype=jnp.float32).reshape(
        num_batches, batch_size, 1
    )

    repertoire = UnstructuredRepertoire.init(
        genotypes=genotypes[0],
        fitnesses=fitnesses[0],
        descriptors=descriptors[0],
        l_value=jnp.array([l_value]),
        max_size=num_batches * batch_size,
    )
    for batch in range(1, num_batches):
        repertoire = repertoire.add(
            batch_of_genotypes=genotypes[batch],
            batch_of_descriptors=descriptors[batch],
            batch_of_fitnesses=fitnesses[batch],
        )

    stored_descriptors = repertoire.descriptors[_stored(repertoire)]
    distances = jnp.linalg.norm(
        stored_descriptors[:, None] - stored_descriptors[None], axis=-1
    )
    num_stored = stored_descriptors.shape[0]
    pairwise_distances = distances[jnp.triu_indices(num_stored, k=1)]
    assert num_stored > 1
    assert jnp.min(pairwise_distances) >= l_value
