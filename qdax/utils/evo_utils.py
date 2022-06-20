# type: ignore
from typing import Any, Callable, Optional, Tuple

import flax
import jax
import jax.numpy as jnp
from qdax.types import Fitness, Genotype, RNGKey
from typing_extensions import TypeAlias


class Solutions(flax.struct.PyTreeNode):
    """Class for keeping track of an item in inventory."""

    genotypes: Genotype
    scores: Fitness

    @property
    def size(self) -> int:
        return len(self.genotypes)

    def save(self, path: str = "./") -> None:
        jnp.save(path + "genotypes.npy", self.genotypes)
        jnp.save(path + "scores.npy", self.scores)


# Define Metrics
MOQDScore: TypeAlias = jnp.ndarray
MaxHypervolume: TypeAlias = jnp.ndarray
MaxScores: TypeAlias = jnp.ndarray
MaxSumScores: TypeAlias = jnp.ndarray
Coverage: TypeAlias = jnp.ndarray
NumSolutions: TypeAlias = jnp.ndarray
GlobalHypervolume: TypeAlias = jnp.ndarray


class MOQDMetrics(flax.struct.PyTreeNode):
    """
    Class to store Multi-Objective QD performance metrics.

        moqd_score: Hypervolume of the Pareto Front in each cell (n_cell, 1)
        max_hypervolume: Maximum hypervolume over every cell (1,)
        max_scores: Maximum values found for each score (n_scores,)
        max_sum_scores: Maximum of sum of scores (1,)
        coverage: Percentage of cells with at least one element
        number_solutions: Total number of solutions
    """

    moqd_score: MOQDScore
    max_hypervolume: MaxHypervolume
    max_scores: MaxScores
    max_sum_scores: MaxSumScores
    coverage: Coverage
    number_solutions: NumSolutions
    global_hypervolume: GlobalHypervolume


def init_population(
    init_genotypes: Genotype,
    scoring_function: Callable[[Genotype], Fitness],
    population_size: Optional[int] = None,
) -> Solutions:
    init_scores = scoring_function(init_genotypes)

    if population_size is not None:
        num_add = population_size - init_genotypes.shape[0]
        if num_add > 0:
            init_genotypes = jnp.concatenate(
                (init_genotypes, jnp.zeros((num_add, init_genotypes.shape[1]))), axis=0
            )
            init_scores = jnp.concatenate(
                (init_scores, -jnp.ones((num_add, init_scores.shape[1])) * jnp.inf),
                axis=0,
            )

    init_solutions = Solutions(genotypes=init_genotypes, scores=init_scores)
    return init_solutions


def sample_in_population(
    solutions: Solutions, random_key: RNGKey, size: int
) -> Tuple[Genotype, RNGKey]:

    key, _ = jax.random.split(random_key)
    mask = solutions.scores != -jnp.inf
    p = jnp.any(mask, axis=-1) / jnp.sum(jnp.any(mask, axis=-1))
    random_sample = jax.random.choice(
        key=key, a=solutions.genotypes, p=p, shape=(size,), replace=False
    )

    return random_sample, key
