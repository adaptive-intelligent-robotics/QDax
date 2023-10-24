"""Core components of the MAP-Elites-sampling algorithm."""
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


@jax.jit
def average(quantities: jnp.ndarray) -> jnp.ndarray:
    """Default expectation extractor using average."""
    return jnp.average(quantities, axis=1)


@jax.jit
def median(quantities: jnp.ndarray) -> jnp.ndarray:
    """Alternative expectation extractor using median.
    More robust to outliers than average."""
    return jnp.median(quantities, axis=1)


@jax.jit
def mode(quantities: jnp.ndarray) -> jnp.ndarray:
    """Alternative expectation extractor using mode.
    More robust to outliers than average.
    WARNING: for multidimensional objects such as descriptor, do
    dimension-wise selection.
    """

    def _mode(quantity: jnp.ndarray) -> jnp.ndarray:

        # Ensure correct dimensions for both single and multi-dimension
        quantity = jnp.reshape(quantity, (quantity.shape[0], -1))

        # Dimension-wise voting in case of multi-dimension
        def _dim_mode(dim_quantity: jnp.ndarray) -> jnp.ndarray:
            unique_vals, counts = jnp.unique(
                dim_quantity, return_counts=True, size=dim_quantity.size
            )
            return unique_vals[jnp.argmax(counts)]

        # vmap over dimensions
        return jnp.squeeze(jax.vmap(_dim_mode)(jnp.transpose(quantity)))

    # vmap over individuals
    return jax.vmap(_mode)(quantities)


@jax.jit
def closest(quantities: jnp.ndarray) -> jnp.ndarray:
    """Alternative expectation extractor selecting individual
    that has the minimum distance to all other individuals. This
    is an approximation of the geometric median.
    More robust to outliers than average."""

    def _closest(values: jnp.ndarray) -> jnp.ndarray:
        def distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return jnp.sqrt(jnp.sum(jnp.square(x - y)))

        distances = jax.vmap(
            jax.vmap(partial(distance), in_axes=(None, 0)), in_axes=(0, None)
        )(values, values)
        return values[jnp.argmin(jnp.mean(distances, axis=0))]

    return jax.vmap(_closest)(quantities)


@jax.jit
def std(quantities: jnp.ndarray) -> jnp.ndarray:
    """Default reproducibility extractor using standard deviation."""
    return jnp.std(quantities, axis=1)


@jax.jit
def mad(quantities: jnp.ndarray) -> jnp.ndarray:
    """Alternative reproducibility extractor using Median Absolute Deviation.
    More robust to outliers than standard deviation."""
    num_samples = quantities.shape[1]
    median = jnp.repeat(
        jnp.median(quantities, axis=1, keepdims=True), num_samples, axis=1
    )
    return jnp.median(jnp.abs(quantities - median), axis=1)


@jax.jit
def iqr(quantities: jnp.ndarray) -> jnp.ndarray:
    """Alternative reproducibility extractor using Inter-Quartile Range.
    More robust to outliers than standard deviation."""
    q1 = jnp.quantile(quantities, 0.25, axis=1)
    q4 = jnp.quantile(quantities, 0.75, axis=1)
    return q4 - q1


@partial(jax.jit, static_argnames=("num_samples",))
def dummy_extra_scores_extractor(
    extra_scores: ExtraScores,
    num_samples: int,
) -> ExtraScores:
    """
    Extract the final extra scores of a policy from multiple samples of
    the same policy in the environment.
    This Dummy implementation just return the full concatenate extra_score
    of all samples without extra computation.

    Args:
        extra_scores: extra scores of the samples
        num_samples: the number of samples used

    Returns:
        the new extra scores after extraction
    """
    return extra_scores


@partial(
    jax.jit,
    static_argnames=(
        "scoring_fn",
        "num_samples",
    ),
)
def multi_sample_scoring_function(
    policies_params: Genotype,
    random_key: RNGKey,
    scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ],
    num_samples: int,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Wrap scoring_function to perform sampling.

    This function returns the fitnesses, descriptors, and extra_scores computed
    over num_samples evaluations with the scoring_fn.

    Args:
        policies_params: policies to evaluate
        random_key: JAX random key
        scoring_fn: scoring function used for evaluation
        num_samples: number of samples to generate for each individual

    Returns:
        (n, num_samples) array of fitnesses,
        (n, num_samples, num_descriptors) array of descriptors,
        dict with num_samples extra_scores per individual,
        JAX random key
    """

    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=num_samples)

    # evaluate
    sample_scoring_fn = jax.vmap(
        scoring_fn,
        # vectorizing over axis 0 vectorizes over the num_samples random keys
        in_axes=(None, 0),
        # indicates that the vectorized axis will become axis 1, i.e., the final
        # output is shape (batch_size, num_samples, ...)
        out_axes=1,
    )
    all_fitnesses, all_descriptors, all_extra_scores, _ = sample_scoring_fn(
        policies_params, keys
    )

    return all_fitnesses, all_descriptors, all_extra_scores, random_key


@partial(
    jax.jit,
    static_argnames=(
        "scoring_fn",
        "num_samples",
        "extra_scores_extractor",
        "fitness_extractor",
        "descriptor_extractor",
    ),
)
def sampling(
    policies_params: Genotype,
    random_key: RNGKey,
    scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ],
    num_samples: int,
    extra_scores_extractor: Callable[
        [ExtraScores, int], ExtraScores
    ] = dummy_extra_scores_extractor,
    fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray] = average,
    descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray] = average,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """Wrap scoring_function to perform sampling.

    This function return the expected fitnesses and descriptors for each
    individual over `num_samples` evaluations using the provided extractor
    function for the fitness and the descriptor.

    Args:
        policies_params: policies to evaluate
        random_key: JAX random key
        scoring_fn: scoring function used for evaluation
        num_samples: number of samples to generate for each individual
        extra_scores_extractor: function to extract the extra_scores from
            multiple samples of the same policy.
        fitness_extractor: function to extract the fitness expectation from
            multiple samples of the same policy.
        descriptor_extractor: function to extract the descriptor expectation
            from multiple samples of the same policy.

    Returns:
        The expected fitnesses, descriptors and extra_scores of the individuals
        A new random key
    """

    # Perform sampling
    (
        all_fitnesses,
        all_descriptors,
        all_extra_scores,
        random_key,
    ) = multi_sample_scoring_function(
        policies_params, random_key, scoring_fn, num_samples
    )

    # Extract final scores
    descriptors = descriptor_extractor(all_descriptors)
    fitnesses = fitness_extractor(all_fitnesses)
    extra_scores = extra_scores_extractor(all_extra_scores, num_samples)

    return fitnesses, descriptors, extra_scores, random_key


@partial(
    jax.jit,
    static_argnames=(
        "scoring_fn",
        "num_samples",
        "extra_scores_extractor",
        "fitness_extractor",
        "descriptor_extractor",
        "fitness_reproducibility_extractor",
        "descriptor_reproducibility_extractor",
    ),
)
def sampling_reproducibility(
    policies_params: Genotype,
    random_key: RNGKey,
    scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ],
    num_samples: int,
    extra_scores_extractor: Callable[
        [ExtraScores, int], ExtraScores
    ] = dummy_extra_scores_extractor,
    fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray] = average,
    descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray] = average,
    fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray] = std,
    descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray] = std,
) -> Tuple[Fitness, Descriptor, ExtraScores, Fitness, Descriptor, RNGKey]:
    """Wrap scoring_function to perform sampling and compute the
    expectation and reproduciblity.

    This function return the reproducibility of fitnesses and descriptors for each
    individual over `num_samples` evaluations using the provided extractor
    function for the fitness and the descriptor.

    Args:
        policies_params: policies to evaluate
        random_key: JAX random key
        scoring_fn: scoring function used for evaluation
        num_samples: number of samples to generate for each individual
        extra_scores_extractor: function to extract the extra_scores from
            multiple samples of the same policy.
        fitness_extractor: function to extract the fitness expectation from
            multiple samples of the same policy.
        descriptor_extractor: function to extract the descriptor expectation
            from multiple samples of the same policy.
        fitness_reproducibility_extractor: function to extract the fitness
            reproducibility from multiple samples of the same policy.
        descriptor_reproducibility_extractor: function to extract the descriptor
            reproducibility from multiple samples of the same policy.

    Returns:
        The expected fitnesses, descriptors and extra_scores of the individuals
        The fitnesses and descriptors reproducibility of the individuals
        A new random key
    """

    # Perform sampling
    (
        all_fitnesses,
        all_descriptors,
        all_extra_scores,
        random_key,
    ) = multi_sample_scoring_function(
        policies_params, random_key, scoring_fn, num_samples
    )

    # Extract final scores
    descriptors = descriptor_extractor(all_descriptors)
    fitnesses = fitness_extractor(all_fitnesses)
    extra_scores = extra_scores_extractor(all_extra_scores, num_samples)

    # Extract reproducibility
    descriptors_reproducibility = descriptor_reproducibility_extractor(all_descriptors)
    fitnesses_reproducibility = fitness_reproducibility_extractor(all_fitnesses)

    return (
        fitnesses,
        descriptors,
        extra_scores,
        fitnesses_reproducibility,
        descriptors_reproducibility,
        random_key,
    )
