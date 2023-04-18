import os

import pytest

from qdax.utils.metrics import CSVLogger


def test_csv_logger() -> None:
    file_location = "test_logs.csv"
    csv_logger = CSVLogger(file_location, ["qd_score", "max_fitness", "coverage"])
    for i in range(10):
        metrics = {"qd_score": i + 0.0, "max_fitness": i + 1.0, "coverage": i + 2.0}
        csv_logger.log(metrics)

    metrics = {"bad_name": 0.0, "max_fitness": 0.0, "coverage": 0.0}
    with pytest.raises(Exception):  # noqa: B017
        csv_logger.log(metrics)

    file_exist = os.path.exists(file_location)
    pytest.assume(file_exist)

    if file_exist:
        os.remove(file_location)
