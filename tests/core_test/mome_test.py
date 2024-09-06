"""Tests that MOME is running."""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import pytest

from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.mutation_operators import (
    polynomial_crossover,
    polynomial_mutation,
)
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.mome import MOME
from qdax.custom_types import Descriptor, ExtraScores, Fitness, RNGKey
from qdax.utils.metrics import default_moqd_metrics


@pytest.mark.parametrize("num_descriptors", [1, 2])
def test_mome(num_descriptors: int) -> None:

    pareto_front_max_length = 50
    num_variables = 120
    num_iterations = 100

    num_descriptors = num_descriptors

    num_centroids = 64
    minval = -2
    maxval = 4
    proportion_to_mutate = 0.6
    eta = 1
    proportion_var_to_change = 0.5
    crossover_percentage = 1.0
    batch_size = 80
    lag = 2.2
    base_lag = 0.0

    def rastrigin_scorer(
        genotypes: jnp.ndarray, base_lag: float, lag: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Rastrigin Scorer with first two dimensions as descriptors
        """
        descriptors = genotypes[:, :num_descriptors]
        f1 = -(
            10 * genotypes.shape[1]
            + jnp.sum(
                (genotypes - base_lag) ** 2
                - 10 * jnp.cos(2 * jnp.pi * (genotypes - base_lag)),
                axis=1,
            )
        )

        f2 = -(
            10 * genotypes.shape[1]
            + jnp.sum(
                (genotypes - lag) ** 2 - 10 * jnp.cos(2 * jnp.pi * (genotypes - lag)),
                axis=1,
            )
        )
        scores = jnp.stack([f1, f2], axis=-1)

        return scores, descriptors

    scoring_function = partial(rastrigin_scorer, base_lag=base_lag, lag=lag)

    def scoring_fn(
        genotypes: jnp.ndarray, random_key: RNGKey
    ) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
        fitnesses, descriptors = scoring_function(genotypes)
        return fitnesses, descriptors, {}, random_key

    reference_point = jnp.array([-150, -150])

    # how to compute metrics from a repertoire
    metrics_function = partial(default_moqd_metrics, reference_point=reference_point)

    # initial population
    random_key = jax.random.PRNGKey(42)
    random_key, subkey = jax.random.split(random_key)
    genotypes = jax.random.uniform(
        subkey,
        (batch_size, num_variables),
        minval=minval,
        maxval=maxval,
        dtype=jnp.float32,
    )

    # crossover function
    crossover_function = partial(
        polynomial_crossover, proportion_var_to_change=proportion_var_to_change
    )

    # mutation function
    mutation_function = partial(
        polynomial_mutation,
        eta=eta,
        minval=minval,
        maxval=maxval,
        proportion_to_mutate=proportion_to_mutate,
    )

    # Define emitter
    mixing_emitter = MixingEmitter(
        mutation_fn=mutation_function,
        variation_fn=crossover_function,
        variation_percentage=crossover_percentage,
        batch_size=batch_size,
    )

    centroids, random_key = compute_cvt_centroids(
        num_descriptors=num_descriptors,
        num_init_cvt_samples=20000,
        num_centroids=num_centroids,
        minval=minval,
        maxval=maxval,
        random_key=random_key,
    )

    mome = MOME(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
    )

    repertoire, emitter_state, random_key = mome.init(
        genotypes, centroids, pareto_front_max_length, random_key
    )

    # Run the algorithm
    (
        repertoire,
        emitter_state,
        random_key,
    ), metrics = jax.lax.scan(
        mome.scan_update,
        (repertoire, emitter_state, random_key),
        (),
        length=num_iterations,
    )

    pytest.assume(metrics["coverage"][-1] > 20)


if __name__ == "__main__":
    test_mome(num_descriptors=1)
