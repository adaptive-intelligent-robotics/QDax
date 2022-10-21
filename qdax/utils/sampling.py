"""Core components of the MAP-Elites-sampling algorithm."""
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


@partial(jax.jit, static_argnames=("num_samples"))
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
        "extra_scores_extractor",
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
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Wrap scoring_function to perform sampling.

    Args:
        policies_params: policies to evaluate
        random_key
        scoring_fn: scoring function used for evaluation
        num_samples
        extra_scores_extractor: function to extract the extra_scores from
            multiple samples of the same policy.

    Returns:
        The average fitness and descriptor of the individuals
        The extra_score extract from samples with extra_scores_extractor
        A new random key
    """

    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=num_samples)

    # evaluate
    sample_scoring_fn = jax.vmap(scoring_fn, (None, 0), 1)
    all_fitnesses, all_descriptors, all_extra_scores, _ = sample_scoring_fn(
        policies_params, keys
    )

    # average results
    descriptors = jnp.average(all_descriptors, axis=1)
    fitnesses = jnp.average(all_fitnesses, axis=1)

    # extract extra scores and add number of evaluations to it
    extra_scores = extra_scores_extractor(all_extra_scores, num_samples)

    return fitnesses, descriptors, extra_scores, random_key
