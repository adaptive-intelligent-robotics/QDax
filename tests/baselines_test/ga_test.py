"""Tests that MOME is running."""

from functools import partial
from typing import Tuple, Type

import jax
import jax.numpy as jnp
import pytest

from qdax.baselines.genetic_algorithm import GeneticAlgorithm
from qdax.baselines.nsga2 import NSGA2
from qdax.baselines.spea2 import SPEA2
from qdax.core.emitters.mutation_operators import (
    polynomial_crossover,
    polynomial_mutation,
)
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.custom_types import ExtraScores, Fitness, RNGKey
from qdax.utils.metrics import default_ga_metrics


@pytest.mark.parametrize("algorithm_class", [GeneticAlgorithm, NSGA2, SPEA2])
def test_ga(algorithm_class: Type[GeneticAlgorithm]) -> None:

    population_size = 1000
    num_iterations = 1000
    proportion_mutation = 0.80
    proportion_var_to_change = 0.5
    proportion_to_mutate = 0.5
    eta = 0.05
    minval, maxval = -5.12, 5.12
    batch_size = 100
    genotype_dim = 6
    lag = 2.2
    base_lag = 0.0
    num_neighbours = 1

    def rastrigin_scorer(
        genotypes: jnp.ndarray, base_lag: float, lag: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Rastrigin Scorer with first two dimensions as descriptors
        """
        descriptors = genotypes[:, :2]
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

    def scoring_fn(genotypes: jnp.ndarray, key: RNGKey) -> Tuple[Fitness, ExtraScores]:
        fitnesses, _ = scoring_function(genotypes)
        return fitnesses, {}

    # initial population
    key = jax.random.key(42)

    key, subkey = jax.random.split(key)
    genotypes = jax.random.uniform(
        subkey,
        (batch_size, genotype_dim),
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
        variation_percentage=1 - proportion_mutation,
        batch_size=batch_size,
    )

    algo_instance = algorithm_class(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=default_ga_metrics,
    )

    key, subkey = jax.random.split(key)
    if isinstance(algo_instance, SPEA2):
        repertoire, emitter_state, init_metrics = algo_instance.init(
            genotypes, population_size, num_neighbours, subkey
        )
    else:
        repertoire, emitter_state, init_metrics = algo_instance.init(
            genotypes, population_size, subkey
        )

    # Run the algorithm
    (
        repertoire,
        emitter_state,
        key,
    ), metrics = jax.lax.scan(
        algo_instance.scan_update,
        (repertoire, emitter_state, key),
        (),
        length=num_iterations,
    )

    x, y = metrics["max_fitness"][-1]
    pytest.assume(x > -20)
    pytest.assume(y > -20)
