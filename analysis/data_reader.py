from typing import List

import numpy as np
import pandas as pd

from qdax.stats.saving_loading_utils import load_dataclass
from qdax.stats.training_state import TrainingState


def read_data(
    file_path: str,
    list_names_components: List[str],
    convert_to_float_numpy_array: bool = True,
) -> dict:
    dict_data_per_component = {name_data: [] for name_data in list_names_components}

    with open(file_path, "r") as file:
        for line in file.readlines():
            list_different_components = filter(bool, line.split("  "))
            list_different_components = list(map(str.strip, list_different_components))
            list_different_components = [
                component.split(" ") for component in list_different_components
            ]

            for index_component, name_component in enumerate(list_names_components):
                dict_data_per_component[name_component].append(
                    list_different_components[index_component]
                )

    try:
        if convert_to_float_numpy_array:
            for name_component in list_names_components:
                dict_data_per_component[name_component] = np.asarray(
                    dict_data_per_component[name_component], dtype=np.float
                )
    except:
        dict_data_per_component = {
            name_data: np.asarray([]).reshape(
                0, len(dict_data_per_component[name_component][0])
            )
            for name_data in list_names_components
        }
    return dict_data_per_component


def read_archive_file(path) -> np.ndarray:
    archives = np.load(path)
    return archives


def read_metrics_csv(path) -> pd.DataFrame:
    df = pd.read_csv(filepath_or_buffer=path)
    return df


def load_training_state(path: str) -> TrainingState:
    return load_dataclass(path)
