import jax.numpy as jnp

from qdax.core.containers.unstructured_repertoire import UnstructuredRepertoire


def test_unstructured_repertoire_add_same_fitness():
    """
    Tests that when a batch of new individuals all share the same descriptors
    and the same fitness, the repertoire does not discard them all.
    """

    # Create a small initial population
    init_batch_size = 3
    genotype_dim = 2
    descriptor_dim = 2

    # Sample data: all individuals have the same descriptor and zero fitness
    init_genotypes = jnp.ones((init_batch_size, genotype_dim))
    init_fitnesses = jnp.zeros((init_batch_size,))
    init_descriptors = jnp.ones((init_batch_size, descriptor_dim))
    init_observations = jnp.zeros((init_batch_size, ))

    # Set the distance threshold and max size
    l_value = jnp.array([0.5])
    max_size = 10

    # Initialize the repertoire
    repertoire = UnstructuredRepertoire.init(
        genotypes=init_genotypes,
        fitnesses=init_fitnesses,
        descriptors=init_descriptors,
        observations=init_observations,
        l_value=l_value,
        max_size=max_size,
    )

    # Verify that the initial repertoire has some individuals
    init_count = jnp.sum(repertoire.fitnesses != -jnp.inf)
    assert init_count > 0, "Initial repertoire should not be empty."

    # Create a new batch: all with same descriptor and same fitness = 0
    new_batch_size = 5
    new_genotypes = jnp.ones((new_batch_size, genotype_dim)) * 2.0
    new_fitnesses = jnp.zeros((new_batch_size,))
    new_descriptors = jnp.ones((new_batch_size, descriptor_dim))  # same as init_descriptors
    new_observations = jnp.zeros((new_batch_size, ))

    # Add new batch to the repertoire
    updated_repertoire = repertoire.add(
        new_genotypes, new_descriptors, new_fitnesses, new_observations
    )

    # Count how many individuals remain
    final_count = jnp.sum(updated_repertoire.fitnesses != -jnp.inf)

    # We check that the repertoire is NOT empty
    # (i.e., the tie-breaking didn't discard all individuals).
    assert final_count > 0, (
        "All individuals were discarded from the repertoire despite tie-breaking!"
    )

    # Optionally, check that the repertoire size does not exceed `max_size`
    assert final_count <= max_size, "Repertoire size exceeded the maximum capacity."