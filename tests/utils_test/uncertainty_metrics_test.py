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
    key = jax.random.key(seed)

    # First, init a deterministic environment
    key, subkey = jax.random.split(key)
    init_policies = jax.random.uniform(
        subkey, shape=(batch_size, genotype_dim), minval=0, maxval=1
    )

    key, subkey = jax.random.split(key)
    fitnesses, descriptors, extra_scores = arm_scoring_function(init_policies, subkey)

    # Initialise a container
    key, subkey = jax.random.split(key)
    centroids = compute_cvt_centroids(
        num_descriptors=2,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=jnp.array([0.0, 0.0]),
        maxval=jnp.array([1.0, 1.0]),
        key=subkey,
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
    key, subkey = jax.random.split(key)
    corrected_repertoire = reevaluation_function(
        repertoire=repertoire,
        empty_corrected_repertoire=empty_corrected_repertoire,
        scoring_fn=arm_scoring_function,
        num_reevals=num_reevals,
        key=subkey,
    )
    pytest.assume(
        jnp.allclose(
            corrected_repertoire.fitnesses, repertoire.fitnesses, rtol=1e-05, atol=1e-05
        )
    )

    # Test that scanned reevaluation_function accurately predicts no change
    key, subkey = jax.random.split(key)
    corrected_repertoire = reevaluation_function(
        repertoire=repertoire,
        empty_corrected_repertoire=empty_corrected_repertoire,
        scoring_fn=arm_scoring_function,
        num_reevals=num_reevals,
        key=subkey,
        scan_size=scan_size,
    )
    pytest.assume(
        jnp.allclose(
            corrected_repertoire.fitnesses, repertoire.fitnesses, rtol=1e-05, atol=1e-05
        )
    )

    # Test that reevaluation_reproducibility_function accurately predicts no change
    key, subkey = jax.random.split(key)
    (
        corrected_repertoire,
        fit_reproducibility_repertoire,
        desc_reproducibility_repertoire,
    ) = reevaluation_reproducibility_function(
        repertoire=repertoire,
        empty_corrected_repertoire=empty_corrected_repertoire,
        scoring_fn=arm_scoring_function,
        num_reevals=num_reevals,
        key=subkey,
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
    key, subkey = jax.random.split(key)
    init_policies = jax.random.uniform(
        subkey, shape=(batch_size, genotype_dim), minval=0, maxval=1
    )
    noisy_scoring_function = functools.partial(
        noisy_arm_scoring_function,
        fit_variance=0.01,
        desc_variance=0.01,
        params_variance=0.0,
    )
    key, subkey = jax.random.split(key)
    fitnesses, descriptors, extra_scores = noisy_scoring_function(init_policies, subkey)

    # Initialise a container
    key, subkey = jax.random.split(key)
    centroids = compute_cvt_centroids(
        num_descriptors=2,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=jnp.array([0.0, 0.0]),
        maxval=jnp.array([1.0, 1.0]),
        key=subkey,
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
    key, subkey = jax.random.split(key)
    (
        corrected_repertoire,
        fit_reproducibility_repertoire,
        desc_reproducibility_repertoire,
    ) = reevaluation_reproducibility_function(
        repertoire=repertoire,
        empty_corrected_repertoire=empty_corrected_repertoire,
        scoring_fn=noisy_scoring_function,
        num_reevals=num_reevals,
        key=subkey,
    )
    pytest.assume(jnp.any(corrected_repertoire.fitnesses > -jnp.inf))
    pytest.assume(jnp.any(fit_reproducibility_repertoire.fitnesses > -jnp.inf))
    pytest.assume(jnp.any(desc_reproducibility_repertoire.fitnesses > -jnp.inf))
