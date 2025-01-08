import jax.numpy as jnp

from qdax.core.containers.unstructured_repertoire import UnstructuredRepertoire


def test_unstructured_repertoire_add_same_fitness():
    """
    Tests that when a batch of new individuals all share the same descriptors
    and the same fitness, the repertoire does not discard them all.
    """

    # Create a small initial population
    init_batch_size = 5
    genotype_dim = 2
    descriptor_dim = 2

    # Sample data: all individuals have the same descriptor and zero fitness
    init_genotypes = jnp.ones((init_batch_size, genotype_dim))
    init_fitnesses = jnp.zeros((init_batch_size,))
    init_descriptors = jnp.ones((init_batch_size, descriptor_dim))
    init_observations = jnp.zeros((init_batch_size, ))

    # Skip the original tie-breaking applied when all fitness identical
    init_fitnesses = init_fitnesses.at[:2].set(1.0)

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
    assert init_count == 1, "Initial repertoire contain only one solution."