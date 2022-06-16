from typing import Tuple

import flax
import jax
import jax.numpy as jnp
from typing_extensions import TypeAlias

from qdax.types import Descriptor, Fitness, Genotype, RNGKey


def compute_pareto_dominance(
    criteria_point: jnp.ndarray, batch_of_criteria: jnp.ndarray
) -> jnp.ndarray:
    """
    Returns if a point is pareto dominated given a set of points or not.
    We use maximization convention here.

    # criteria_point of shape (num_criteria,)
    # batch_of_criteria of shape (num_points, num_criteria)
    """
    diff = jnp.subtract(batch_of_criteria, criteria_point)
    return jnp.any(jnp.all(diff > 0, axis=-1))


def compute_pareto_front(batch_of_criteria: jnp.ndarray) -> jnp.ndarray:
    """
    Returns an array of boolean that states for each element if it is in
    the pareto front or not.

    # batch_of_criteria of shape (num_points, num_criteria)
    """
    func = jax.vmap(lambda x: ~compute_pareto_dominance(x, batch_of_criteria))
    return func(batch_of_criteria)


def compute_masked_pareto_dominance(
    criteria_point: jnp.ndarray, batch_of_criteria: jnp.ndarray, mask: jnp.ndarray
) -> jnp.ndarray:
    """
    Returns if a point is pareto dominated given a set of points or not.
    We use maximization convention here.
    This function is to be used with constant size batches of criteria,
    thus a mask is used to know which values are padded.

    # criteria_point of shape (num_criteria,)
    # batch_of_criteria of shape (batch_size, num_criteria)
    # mask of shape (batch_size,), 1.0 where there is not element, 0 otherwise
    """

    diff = jnp.subtract(batch_of_criteria, criteria_point)
    neutral_values = -jnp.ones_like(diff)
    diff = jax.vmap(lambda x1, x2: jnp.where(mask, x1, x2), in_axes=(1, 1), out_axes=1)(
        neutral_values, diff
    )
    return jnp.any(jnp.all(diff > 0, axis=-1))


def compute_masked_pareto_front(
    batch_of_criteria: jnp.ndarray, mask: jnp.ndarray
) -> jnp.ndarray:
    """
    Returns an array of boolean that states for each element if it is to be
    considered or not. This function works is to be used constant size batches of
    criteria, thus a mask is used to know which values are padded.
    """
    func = jax.vmap(
        lambda x: ~compute_masked_pareto_dominance(x, batch_of_criteria, mask)
    )
    return func(batch_of_criteria) * ~mask


def sample_in_masked_pareto_front(
    pareto_front_genotypes: Genotype,
    mask: jnp.ndarray,
    num_samples: int,
    random_key: RNGKey,
) -> Genotype:
    """
    Sample num_samples elements in masked pareto front.

    Note: do not retrieve a random key because this function
    is to be vmapped. The public method that uses this function
    will return a random key
    """
    p = (1.0 - mask) / jnp.sum(1.0 - mask)

    genotype_sample = jax.tree_map(
        lambda x: jax.random.choice(random_key, x, shape=(num_samples,), p=p),
        pareto_front_genotypes,
    )

    return genotype_sample


def update_masked_pareto_front(
    pareto_front_fitness: Fitness,
    pareto_front_genotypes: Genotype,
    pareto_front_descriptors: Descriptor,
    mask: jnp.ndarray,
    new_batch_of_criteria: Fitness,
    new_batch_of_genotypes: Genotype,
    new_batch_of_descriptors: Descriptor,
    new_mask: jnp.ndarray,
) -> Tuple[Fitness, Genotype, Descriptor, jnp.ndarray]:
    """
    Takes a fixed size pareto front, its mask and new points to add.
    Returns updated front and mask.
    """
    # get dimensions
    batch_size = new_batch_of_criteria.shape[0]
    num_criteria = new_batch_of_criteria.shape[1]

    pareto_front_len = pareto_front_fitness.shape[0]

    first_leaf = jax.tree_leaves(new_batch_of_genotypes)[0]
    genotypes_dim = first_leaf.shape[1]

    descriptors_dim = new_batch_of_descriptors.shape[1]

    # gather all data
    cat_mask = jnp.concatenate([mask, new_mask], axis=-1)
    cat_f = jnp.concatenate([pareto_front_fitness, new_batch_of_criteria], axis=0)
    cat_genotypes = jax.tree_map(
        lambda x, y: jnp.concatenate([x, y], axis=0),
        pareto_front_genotypes,
        new_batch_of_genotypes,
    )
    cat_descriptors = jnp.concatenate(
        [pareto_front_descriptors, new_batch_of_descriptors], axis=0
    )

    # get new front
    cat_bool_front = compute_masked_pareto_front(batch_of_criteria=cat_f, mask=cat_mask)

    # get corresponding indices
    indices = jnp.arange(start=0, stop=pareto_front_len + batch_size) * cat_bool_front
    indices = indices + ~cat_bool_front * (batch_size + pareto_front_len - 1)
    indices = jnp.sort(indices)

    # get new fitness, genotypes and descriptors
    new_front_fitness = jnp.take(cat_f, indices, axis=0)
    new_front_genotypes = jax.tree_map(
        lambda x: jnp.take(x, indices, axis=0), cat_genotypes
    )
    new_front_descriptors = jnp.take(cat_descriptors, indices, axis=0)

    # compute new mask
    num_front_elements = jnp.sum(cat_bool_front)
    new_mask_indices = jnp.arange(start=0, stop=batch_size + pareto_front_len)
    new_mask_indices = (num_front_elements - new_mask_indices) > 0

    new_mask = jnp.where(
        new_mask_indices,
        jnp.ones(shape=batch_size + pareto_front_len, dtype=bool),
        jnp.zeros(shape=batch_size + pareto_front_len, dtype=bool),
    )

    fitness_mask = jnp.repeat(jnp.expand_dims(new_mask, axis=-1), num_criteria, axis=-1)
    new_front_fitness = new_front_fitness * fitness_mask
    new_front_fitness = new_front_fitness[: len(pareto_front_fitness), :]

    genotypes_mask = jnp.repeat(
        jnp.expand_dims(new_mask, axis=-1), genotypes_dim, axis=-1
    )
    new_front_genotypes = jax.tree_map(
        lambda x: x * genotypes_mask, new_front_genotypes
    )
    new_front_genotypes = jax.tree_map(
        lambda x: x[: len(pareto_front_fitness), :], new_front_genotypes
    )

    descriptors_mask = jnp.repeat(
        jnp.expand_dims(new_mask, axis=-1), descriptors_dim, axis=-1
    )
    new_front_descriptors = new_front_descriptors * descriptors_mask
    new_front_descriptors = new_front_descriptors[: len(pareto_front_fitness), :]

    new_mask = ~new_mask[: len(pareto_front_fitness)]

    return new_front_fitness, new_front_genotypes, new_front_descriptors, new_mask


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


def compute_hypervolume(
    pareto_front: jnp.ndarray, reference_point: jnp.ndarray
) -> jnp.ndarray:
    """Compute hypervolume of a pareto front.

    TODO: implement recursive version of
    https://github.com/anyoptimization/pymoo/blob/master/pymoo/vendor/hv.py
    """

    num_objectives = pareto_front.shape[1]

    assert (
        num_objectives == 2
    ), "Hypervolume calculation for more than 2 objectives not yet supported."

    pareto_front = jnp.concatenate(  # type: ignore
        (pareto_front, jnp.expand_dims(reference_point, axis=0)), axis=0
    )
    idx = jnp.argsort(pareto_front[:, 0])
    mask = pareto_front[idx, :] != -jnp.inf
    sorted_front = (pareto_front[idx, :] - reference_point) * mask
    sumdiff = (sorted_front[1:, 0] - sorted_front[:-1, 0]) * sorted_front[1:, 1]
    sumdiff = sumdiff * mask[:-1, 0]
    hypervolume = jnp.sum(sumdiff)

    return hypervolume
