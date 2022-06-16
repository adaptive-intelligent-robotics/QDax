from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.core.map_elites import MAPElites
from qdax.types import Centroid, RNGKey
from qdax.utils.mome_utils import (
    MOQDMetrics,
    compute_hypervolume,
    compute_masked_pareto_front,
)


class MOME(MAPElites):
    """Implements Multi-Objectives MAP Elites.

    Note: most functions are inherited from MAPElites.

    Args:
        MAPElites: _description_

    Returns:
        _description_
    """

    @partial(jax.jit, static_argnames=("self", "pareto_front_max_length"))
    def init(
        self,
        init_genotypes: jnp.ndarray,
        centroids: Centroid,
        pareto_front_max_length: int,
        random_key: RNGKey,
    ) -> Tuple[MOMERepertoire, Optional[EmitterState], RNGKey]:
        """
        Initialize a MOME grid with an initial population of genotypes. Requires
        the definition of centroids that can be computed with any method such as
        CVT or Euclidean mapping.
        """

        # first score
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            init_genotypes, random_key
        )

        # init the repertoire
        repertoire = MOMERepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            pareto_front_max_length=pareto_front_max_length,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return repertoire, emitter_state, random_key


def compute_moqd_metrics(
    grid: MOMERepertoire, reference_point: jnp.ndarray
) -> MOQDMetrics:
    """
    Compute the MOQD metric given a MOME grid and a reference point.
    """
    grid_empty = grid.fitnesses == -jnp.inf
    grid_empty = jnp.all(grid_empty, axis=-1)
    grid_not_empty = ~grid_empty
    grid_not_empty = jnp.any(grid_not_empty, axis=-1)
    coverage = 100 * jnp.mean(grid_not_empty)
    hypervolume_function = partial(compute_hypervolume, reference_point=reference_point)
    moqd_scores = jax.vmap(hypervolume_function)(grid.fitnesses)
    moqd_scores = jnp.where(grid_not_empty, moqd_scores, -jnp.inf)
    max_hypervolume = jnp.max(moqd_scores)
    max_scores = jnp.max(grid.fitnesses, axis=(0, 1))
    max_sum_scores = jnp.max(jnp.sum(grid.fitnesses, axis=-1), axis=(0, 1))
    num_solutions = jnp.sum(~grid_empty)
    (
        pareto_front,
        _,
    ) = compute_global_pareto_front(grid)

    global_hypervolume = compute_hypervolume(
        pareto_front, reference_point=reference_point
    )
    metrics = MOQDMetrics(
        moqd_score=moqd_scores,
        max_hypervolume=max_hypervolume,
        max_scores=max_scores,
        max_sum_scores=max_sum_scores,
        coverage=coverage,
        number_solutions=num_solutions,
        global_hypervolume=global_hypervolume,
    )

    return metrics


def compute_global_pareto_front(
    grid: MOMERepertoire,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Merge all the pareto fronts of the MOME grid into a single one called global.
    """
    scores = jnp.concatenate(grid.fitnesses, axis=0)
    mask = jnp.any(scores == -jnp.inf, axis=-1)
    pareto_bool = compute_masked_pareto_front(scores, mask)
    pareto_front = scores - jnp.inf * (~jnp.array([pareto_bool, pareto_bool]).T)

    return pareto_front, pareto_bool


def add_init_metrics(metrics: MOQDMetrics, init_metrics: MOQDMetrics) -> MOQDMetrics:
    """
    Append an initial metric to the run metrics.
    """
    metrics = jax.tree_multimap(
        lambda x, y: jnp.concatenate((jnp.expand_dims(x, axis=0), y), axis=0),
        init_metrics,
        metrics,
    )
    return metrics
