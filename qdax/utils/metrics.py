"""Defines functions to retrieve metrics from training processes."""

from __future__ import annotations

import csv
from functools import partial
from typing import Dict, List

import jax
from jax import numpy as jnp

from qdax.core.containers.mome_repertoire import (
    MOMERepertoire,
    compute_global_pareto_front,
)
from qdax.core.containers.repertoire import MapElitesRepertoire
from qdax.utils.mome_utils import compute_hypervolume


class CSVLogger:
    """Logger to save metrics of an experiment in a csv file
    during the training process.
    """

    def __init__(self, filename: str, header: List) -> None:
        """Create the csv logger, create a file and write the
        header.

        Args:
            filename: path to which the file will be saved.
            header: header of the csv file.
        """
        self._filename = filename
        self._header = header
        with open(self._filename, "w") as file:
            writer = csv.DictWriter(file, fieldnames=self._header)
            # write the header
            writer.writeheader()

    def log(self, metrics: Dict[str, float]) -> None:
        """Log new metrics to the csv file.

        Args:
            metrics: A dictionary containing the metrics that
                need to be saved.
        """
        with open(self._filename, "a") as file:
            writer = csv.DictWriter(file, fieldnames=self._header)
            # write new metrics in a raw
            writer.writerow(metrics)


def default_qd_metrics(
    repertoire: MapElitesRepertoire, qd_offset: float
) -> Dict[str, jnp.ndarray]:
    """Compute the usual QD metrics that one can retrieve
    from a MAP Elites repertoire.

    Args:
        repertoire: a MAP-Elites repertoire
        qd_offset: an offset used to ensure that the QD score
            will be positive and increasing with the number
            of individuals.

    Returns:
        a dictionary containing the QD score (sum of fitnesses
            modified to be all positive), the max fitness of the
            repertoire, the coverage (number of niche filled in
            the repertoire).
    """

    # get metrics
    grid_empty = repertoire.fitnesses == -jnp.inf
    qd_score = jnp.sum(repertoire.fitnesses, where=~grid_empty)
    qd_score += qd_offset * jnp.sum(1.0 - grid_empty)
    coverage = 100 * jnp.mean(1.0 - grid_empty)
    max_fitness = jnp.max(repertoire.fitnesses)

    return {"qd_score": qd_score, "max_fitness": max_fitness, "coverage": coverage}


def compute_moqd_metrics(
    grid: MOMERepertoire, reference_point: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
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
    metrics = {
        "moqd_score": moqd_scores,
        "max_hypervolume": max_hypervolume,
        "max_scores": max_scores,
        "max_sum_scores": max_sum_scores,
        "coverage": coverage,
        "number_solutions": num_solutions,
        "global_hypervolume": global_hypervolume,
    }

    return metrics
