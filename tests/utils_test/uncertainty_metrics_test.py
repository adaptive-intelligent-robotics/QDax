import functools

import jax
import jax.numpy as jnp
import pytest

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from qdax.tasks.arm import arm_scoring_function, noisy_arm_scoring_function
from qdax.utils.uncertainty_metrics import (
    reevaluation_function,
    reevaluation_reproducibility_function,
)


def test_uncertainty_metrics() -> None:
    seed = 42
    num_reevals = 512
    scan_size = 128
    batch_size = 512
    num_init_cvt_samples = 50000
    num_centroids = 1024
    genotype_dim = 8

    # Init a random key
    random_key = jax.random.PRNGKey(seed)

    # First, init a deterministic environment
    init_policies = jax.random.uniform(
        random_key, shape=(batch_size, genotype_dim), minval=0, maxval=1
    )
    fitnesses, descriptors, extra_scores, random_key = arm_scoring_function(
        init_policies, random_key
    )

    # Initialise a container
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=2,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=jnp.array([0.0, 0.0]),
        maxval=jnp.array([1.0, 1.0]),
        random_key=random_key,
    )
    repertoire = MapElitesRepertoire.init(
        genotypes=init_policies,
        fitnesses=fitnesses,
        descriptors=descriptors,
        centroids=centroids,
        extra_scores=extra_scores,
    )

    # Initialise an empty container for corrected repertoire
    fitnesses = jnp.full_like(fitnesses, -jnp.inf)
    empty_corrected_repertoire = MapElitesRepertoire.init(
        genotypes=init_policies,
        fitnesses=fitnesses,
        descriptors=descriptors,
        centroids=centroids,
        extra_scores=extra_scores,
    )

    # Test that reevaluation_function accurately predicts no change
    corrected_repertoire, random_key = reevaluation_function(
        repertoire=repertoire,
        empty_corrected_repertoire=empty_corrected_repertoire,
        scoring_fn=arm_scoring_function,
        num_reevals=num_reevals,
        random_key=random_key,
    )
    pytest.assume(
        jnp.allclose(
            corrected_repertoire.fitnesses, repertoire.fitnesses, rtol=1e-05, atol=1e-05
        )
    )

    # Test that scanned reevaluation_function accurately predicts no change
    corrected_repertoire, random_key = reevaluation_function(
        repertoire=repertoire,
        empty_corrected_repertoire=empty_corrected_repertoire,
        scoring_fn=arm_scoring_function,
        num_reevals=num_reevals,
        random_key=random_key,
        scan_size=scan_size,
    )
    pytest.assume(
        jnp.allclose(
            corrected_repertoire.fitnesses, repertoire.fitnesses, rtol=1e-05, atol=1e-05
        )
    )

    # Test that reevaluation_reproducibility_function accurately predicts no change
    (
        corrected_repertoire,
        fit_reproducibility_repertoire,
        desc_reproducibility_repertoire,
        random_key,
    ) = reevaluation_reproducibility_function(
        repertoire=repertoire,
        empty_corrected_repertoire=empty_corrected_repertoire,
        scoring_fn=arm_scoring_function,
        num_reevals=num_reevals,
        random_key=random_key,
    )
    pytest.assume(
        jnp.allclose(
            corrected_repertoire.fitnesses, repertoire.fitnesses, rtol=1e-05, atol=1e-05
        )
    )
    zero_fitnesses = jnp.where(
        repertoire.fitnesses > -jnp.inf,
        0.0,
        -jnp.inf,
    )
    pytest.assume(
        jnp.allclose(
            fit_reproducibility_repertoire.fitnesses,
            zero_fitnesses,
            rtol=1e-05,
            atol=1e-05,
        )
    )
    pytest.assume(
        jnp.allclose(
            desc_reproducibility_repertoire.fitnesses,
            zero_fitnesses,
            rtol=1e-05,
            atol=1e-05,
        )
    )

    # Second, init a stochastic environment
    init_policies = jax.random.uniform(
        random_key, shape=(batch_size, genotype_dim), minval=0, maxval=1
    )
    noisy_scoring_function = functools.partial(
        noisy_arm_scoring_function,
        fit_variance=0.01,
        desc_variance=0.01,
        params_variance=0.0,
    )
    fitnesses, descriptors, extra_scores, random_key = noisy_scoring_function(
        init_policies, random_key
    )

    # Initialise a container
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=2,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=jnp.array([0.0, 0.0]),
        maxval=jnp.array([1.0, 1.0]),
        random_key=random_key,
    )
    repertoire = MapElitesRepertoire.init(
        genotypes=init_policies,
        fitnesses=fitnesses,
        descriptors=descriptors,
        centroids=centroids,
        extra_scores=extra_scores,
    )

    # Initialise an empty container for corrected repertoire
    fitnesses = jnp.full_like(fitnesses, -jnp.inf)
    empty_corrected_repertoire = MapElitesRepertoire.init(
        genotypes=init_policies,
        fitnesses=fitnesses,
        descriptors=descriptors,
        centroids=centroids,
        extra_scores=extra_scores,
    )

    # Test that reevaluation_function runs and keeps at least one solution
    (
        corrected_repertoire,
        fit_reproducibility_repertoire,
        desc_reproducibility_repertoire,
        random_key,
    ) = reevaluation_reproducibility_function(
        repertoire=repertoire,
        empty_corrected_repertoire=empty_corrected_repertoire,
        scoring_fn=noisy_scoring_function,
        num_reevals=num_reevals,
        random_key=random_key,
    )
    pytest.assume(jnp.any(corrected_repertoire.fitnesses > -jnp.inf))
    pytest.assume(jnp.any(fit_reproducibility_repertoire.fitnesses > -jnp.inf))
    pytest.assume(jnp.any(desc_reproducibility_repertoire.fitnesses > -jnp.inf))
