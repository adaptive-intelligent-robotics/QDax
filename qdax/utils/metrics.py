from __future__ import annotations

import csv
from typing import Dict, List

from jax import numpy as jnp

from qdax.core.containers.repertoire import MapElitesRepertoire


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
        self._file = open(self._filename, "w")
        self._writer = csv.DictWriter(self._file, fieldnames=header)

        # write the header
        self._writer.writeheader()

    def log(self, metrics: Dict[str, float]) -> None:
        """Log new metrics to the csv file.

        Args:
            metrics: A dictionary containing the metrics that
                need to be saved.
        """
        self._writer.writerow(metrics)

    def close(self) -> None:
        """Close the file."""
        self._file.close()


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
