from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from qdax.utils.sampling import (
    dummy_extra_scores_extractor,
    median,
    multi_sample_scoring_function,
    std,
)


@partial(
    jax.jit,
    static_argnames=(
        "scoring_fn",
        "num_reevals",
        "fitness_extractor",
        "descriptor_extractor",
        "extra_scores_extractor",
        "scan_size",
    ),
)
def reevaluation_function(
    repertoire: MapElitesRepertoire,
    key: RNGKey,
    empty_corrected_repertoire: MapElitesRepertoire,
    scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores],
    ],
    num_reevals: int,
    fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray] = median,
    descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray] = median,
    extra_scores_extractor: Callable[
        [ExtraScores, int], ExtraScores
    ] = dummy_extra_scores_extractor,
    scan_size: int = 0,
) -> MapElitesRepertoire:
    """
    Perform reevaluation of a repertoire and construct a corrected repertoire from it.

    Args:
        repertoire: repertoire to reevaluate.
        empty_corrected_repertoire: repertoire to be filled with reevaluated solutions,
            allow to use a different type of repertoire than the one from the algorithm.
        key: JAX random key.
        scoring_fn: scoring function used for evaluation.
        num_reevals: number of samples to generate for each individual.
        fitness_extractor: function to extract the final fitness from
            multiple samples of the same solution (default: median).
        descriptor_extractor: function to extract the final descriptor from
            multiple samples of the same solution (default: median).
        extra_scores_extractor: function to extract the extra_scores from
            multiple samples of the same solution (default: no effect).
        scan_size: allow to split the reevaluations in multiple batch to reduce
            the memory load of the reevaluation.
    Returns:
        The corrected repertoire.
    """

    # If no reevaluations, return copies of the original container
    if num_reevals == 0:
        return repertoire

    # Perform reevaluation
    key, subkey = jax.random.split(key)
    (
        all_fitnesses,
        all_descriptors,
        all_extra_scores,
    ) = _perform_reevaluation(
        policies_params=repertoire.genotypes,
        key=subkey,
        scoring_fn=scoring_fn,
        num_reevals=num_reevals,
        scan_size=scan_size,
    )

    # Extract the final scores
    extra_scores = extra_scores_extractor(all_extra_scores, num_reevals)
    fitnesses = fitness_extractor(all_fitnesses)
    descriptors = descriptor_extractor(all_descriptors)

    # Set -inf fitness for all unexisting indivs
    fitnesses = jnp.where(repertoire.fitnesses == -jnp.inf, -jnp.inf, fitnesses)

    # Fill-in the corrected repertoire
    corrected_repertoire = empty_corrected_repertoire.add(
        batch_of_genotypes=repertoire.genotypes,
        batch_of_descriptors=descriptors,
        batch_of_fitnesses=fitnesses,
        batch_of_extra_scores=extra_scores,
    )

    return corrected_repertoire  # type: ignore


@partial(
    jax.jit,
    static_argnames=(
        "scoring_fn",
        "num_reevals",
        "fitness_extractor",
        "fitness_reproducibility_extractor",
        "descriptor_extractor",
        "descriptor_reproducibility_extractor",
        "extra_scores_extractor",
        "scan_size",
    ),
)
def reevaluation_reproducibility_function(
    repertoire: MapElitesRepertoire,
    key: RNGKey,
    empty_corrected_repertoire: MapElitesRepertoire,
    scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores],
    ],
    num_reevals: int,
    fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray] = median,
    fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray] = std,
    descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray] = median,
    descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray] = std,
    extra_scores_extractor: Callable[
        [ExtraScores, int], ExtraScores
    ] = dummy_extra_scores_extractor,
    scan_size: int = 0,
) -> Tuple[MapElitesRepertoire, MapElitesRepertoire, MapElitesRepertoire]:
    """
    Perform reevaluation of a repertoire and construct a corrected repertoire and a
    reproducibility repertoire from it.

    Args:
        repertoire: repertoire to reevaluate.
        empty_corrected_repertoire: repertoire to be filled with reevaluated solutions,
            allow to use a different type of repertoire than the one from the algorithm.
        key: JAX random key.
        scoring_fn: scoring function used for evaluation.
        num_reevals: number of samples to generate for each individual.
        fitness_extractor: function to extract the final fitness from
            multiple samples of the same solution (default: median).
        fitness_reproducibility_extractor: function to extract the fitness
            reproducibility from multiple samples of the same solution (default: std).
        descriptor_extractor: function to extract the final descriptor from
            multiple samples of the same solution (default: median).
        descriptor_reproducibility_extractor: function to extract the descriptor
            reproducibility from multiple samples of the same solution (default: std).
        extra_scores_extractor: function to extract the extra_scores from
            multiple samples of the same solution (default: no effect).
        scan_size: allow to split the reevaluations in multiple batch to reduce
            the memory load of the reevaluation.
    Returns:
        The corrected repertoire.
        A repertoire storing reproducibility in fitness.
        A repertoire storing reproducibility in descriptor.
    """

    # If no reevaluations, return copies of the original container
    if num_reevals == 0:
        return (
            repertoire,
            repertoire,
            repertoire,
        )

    # Perform reevaluation
    (
        all_fitnesses,
        all_descriptors,
        all_extra_scores,
    ) = _perform_reevaluation(
        policies_params=repertoire.genotypes,
        key=key,
        scoring_fn=scoring_fn,
        num_reevals=num_reevals,
        scan_size=scan_size,
    )

    # Extract the final scores
    extra_scores = extra_scores_extractor(all_extra_scores, num_reevals)
    fitnesses = fitness_extractor(all_fitnesses)
    fitnesses_reproducibility = fitness_reproducibility_extractor(all_fitnesses)
    descriptors = descriptor_extractor(all_descriptors)
    descriptors_reproducibility = descriptor_reproducibility_extractor(all_descriptors)

    # WARNING: in the case of descriptors_reproducibility, take average over dimensions
    descriptors_reproducibility = jnp.average(descriptors_reproducibility, axis=-1)

    # Set -inf fitness for all unexisting indivs
    fitnesses = jnp.where(repertoire.fitnesses == -jnp.inf, -jnp.inf, fitnesses)
    fitnesses_reproducibility = jnp.where(
        repertoire.fitnesses == -jnp.inf, -jnp.inf, fitnesses_reproducibility
    )
    descriptors_reproducibility = jnp.where(
        repertoire.fitnesses == -jnp.inf,
        -jnp.inf,
        jnp.expand_dims(descriptors_reproducibility, axis=-1),
    )

    # Fill-in corrected repertoire
    corrected_repertoire = empty_corrected_repertoire.add(
        batch_of_genotypes=repertoire.genotypes,
        batch_of_descriptors=descriptors,
        batch_of_fitnesses=fitnesses,
        batch_of_extra_scores=extra_scores,
    )

    # Fill-in fit_reproducibility repertoire
    fit_reproducibility_repertoire = empty_corrected_repertoire.add(
        batch_of_genotypes=repertoire.genotypes,
        batch_of_descriptors=repertoire.descriptors,
        batch_of_fitnesses=fitnesses_reproducibility,
        batch_of_extra_scores=extra_scores,
    )

    # Fill-in desc_reproducibility repertoire
    desc_reproducibility_repertoire = empty_corrected_repertoire.add(
        batch_of_genotypes=repertoire.genotypes,
        batch_of_descriptors=repertoire.descriptors,
        batch_of_fitnesses=descriptors_reproducibility,
        batch_of_extra_scores=extra_scores,
    )

    return (
        corrected_repertoire,
        fit_reproducibility_repertoire,
        desc_reproducibility_repertoire,
    )


def _perform_reevaluation(
    policies_params: Genotype,
    key: RNGKey,
    scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores],
    ],
    num_reevals: int,
    scan_size: int = 0,
) -> Tuple[Fitness, Descriptor, ExtraScores]:
    """
    Sub-function used to perform reevaluation of a repertoire in uncertain applications.

    Args:
        policies_params: genotypes to reevaluate.
        key: JAX random key.
        scoring_fn: scoring function used for evaluation.
        num_reevals: number of samples to generate for each individual.
        scan_size: allow to split the reevaluations in multiple batch to reduce
            the memory load of the reevaluation.
    Returns:
        The fitnesses, descriptors and extra score from the reevaluation.
    """

    # If no need for scan, call the sampling function
    if scan_size == 0:
        key, subkey = jax.random.split(key)
        (
            all_fitnesses,
            all_descriptors,
            all_extra_scores,
        ) = multi_sample_scoring_function(
            policies_params=policies_params,
            key=subkey,
            scoring_fn=scoring_fn,
            num_samples=num_reevals,
        )

    # If need for scan, call the sampling function multiple times
    else:

        # Ensure that num_reevals is a multiple of scan_size
        assert (
            num_reevals % scan_size == 0
        ), "num_reevals should be a multiple of scan_size to be able to scan."
        num_loops = num_reevals // scan_size

        def _sampling_scan(
            key: RNGKey,
            _: Tuple[()],
        ) -> Tuple[Tuple[RNGKey], Tuple[Fitness, Descriptor, ExtraScores]]:
            key, subkey = jax.random.split(key)
            (
                all_fitnesses,
                all_descriptors,
                all_extra_scores,
            ) = multi_sample_scoring_function(
                policies_params=policies_params,
                key=subkey,
                scoring_fn=scoring_fn,
                num_samples=scan_size,
            )
            return (key), (
                all_fitnesses,
                all_descriptors,
                all_extra_scores,
            )

        (key), (
            all_fitnesses,
            all_descriptors,
            all_extra_scores,
        ) = jax.lax.scan(_sampling_scan, (key), (), length=num_loops)
        all_fitnesses = jnp.hstack(all_fitnesses)
        all_descriptors = jnp.hstack(all_descriptors)

    all_fitnesses = all_fitnesses.reshape(all_fitnesses.shape[0], num_reevals, -1)
    return all_fitnesses, all_descriptors, all_extra_scores
