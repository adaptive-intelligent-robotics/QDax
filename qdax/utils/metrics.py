"""Defines functions to retrieve metrics from training processes."""

from __future__ import annotations

import csv
from functools import partial
from typing import Dict, List, Optional

import jax
from jax import numpy as jnp

from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.core.containers.mtmb_repertoire import MTMBRepertoire
from qdax.custom_types import Metrics
from qdax.utils.pareto_front import compute_hypervolume


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


def default_ga_metrics(
    repertoire: GARepertoire,
) -> Metrics:
    """Compute the usual GA metrics that one can retrieve
    from a GA repertoire.

    Args:
        repertoire: a GA repertoire

    Returns:
        a dictionary containing the max fitness of the
            repertoire.
    """

    # get metrics
    max_fitness = jnp.max(repertoire.fitnesses, axis=0)

    return {
        "max_fitness": max_fitness,
    }


def default_qd_metrics(repertoire: MapElitesRepertoire, qd_offset: float) -> Metrics:
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
    repertoire_empty = repertoire.fitnesses == -jnp.inf
    qd_score = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
    qd_score += qd_offset * jnp.sum(1.0 - repertoire_empty)
    coverage = 100 * jnp.mean(1.0 - repertoire_empty)
    max_fitness = jnp.max(repertoire.fitnesses)

    return {"qd_score": qd_score, "max_fitness": max_fitness, "coverage": coverage}


def default_mtmb_metrics(
    repertoire: MTMBRepertoire,
    qd_offset: float,
    fitness_threshold: Optional[float] = None,
) -> Metrics:
    """Compute the usual QD metrics as well as multi-task metrics from a
    Multi-Task Multi-Behavior MAP-Elites repertoire.

    Args:
        repertoire: an MTMB repertoire
        qd_offset: an offset used to ensure that the QD score
            will be positive and increasing with the number
            of individuals.
        fitness_threshold: an optional fitness value above which an elite is
            considered a solution of its task, following the MTMB MAP-Elites
            paper. When provided, the percentage of solved tasks and the
            number of solutions per solved task are also computed.

    Returns:
        a dictionary containing the QD score (sum of fitnesses modified to
            be all positive), the max fitness of the repertoire, the
            coverage (fraction of cells filled in the repertoire), the task
            coverage (fraction of tasks with at least one elite) and the
            mean number of elites of the covered tasks.
    """
    num_tasks = repertoire.task_descriptors.shape[0]
    num_centroids = repertoire.centroids.shape[0]

    # get standard QD metrics
    repertoire_empty = repertoire.fitnesses == -jnp.inf
    qd_score = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
    qd_score += qd_offset * jnp.sum(1.0 - repertoire_empty)
    coverage = 100 * jnp.mean(1.0 - repertoire_empty)
    max_fitness = jnp.max(repertoire.fitnesses)

    # get multi-task metrics
    occupancy = (~repertoire_empty[:, 0]).reshape(num_tasks, num_centroids)
    task_has_elite = jnp.any(occupancy, axis=1)
    task_coverage = 100 * jnp.mean(task_has_elite)
    elites_per_covered_task = jnp.sum(occupancy) / jnp.maximum(
        jnp.sum(task_has_elite), 1
    )

    metrics = {
        "qd_score": qd_score,
        "max_fitness": max_fitness,
        "coverage": coverage,
        "task_coverage": task_coverage,
        "elites_per_covered_task": elites_per_covered_task,
    }

    if fitness_threshold is not None:
        solutions = (repertoire.fitnesses[:, 0] >= fitness_threshold).reshape(
            num_tasks, num_centroids
        )
        task_solved = jnp.any(solutions, axis=1)
        metrics["tasks_solved"] = 100 * jnp.mean(task_solved)
        metrics["solutions_per_solved_task"] = jnp.sum(solutions) / jnp.maximum(
            jnp.sum(task_solved), 1
        )

    return metrics


def default_moqd_metrics(
    repertoire: MOMERepertoire, reference_point: jax.Array
) -> Metrics:
    """Compute the MOQD metric given a MOME repertoire and a reference point.

    Args:
        repertoire: a MOME repertoire.
        reference_point: the hypervolume of a pareto front has to be computed
            relatively to a reference point.

    Returns:
        A dictionary containing all the computed metrics.
    """
    repertoire_empty = repertoire.fitnesses == -jnp.inf
    repertoire_empty = jnp.all(repertoire_empty, axis=-1)
    repertoire_not_empty = ~repertoire_empty
    repertoire_not_empty = jnp.any(repertoire_not_empty, axis=-1)
    coverage = 100 * jnp.mean(repertoire_not_empty)
    hypervolume_function = partial(compute_hypervolume, reference_point=reference_point)
    moqd_scores = jax.vmap(hypervolume_function)(repertoire.fitnesses)
    moqd_scores = jnp.where(repertoire_not_empty, moqd_scores, -jnp.inf)
    max_hypervolume = jnp.max(moqd_scores)
    max_scores = jnp.max(repertoire.fitnesses, axis=(0, 1))
    max_sum_scores = jnp.max(jnp.sum(repertoire.fitnesses, axis=-1), axis=(0, 1))
    num_solutions = jnp.sum(~repertoire_empty)
    (
        pareto_front,
        _,
    ) = repertoire.compute_global_pareto_front()

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
