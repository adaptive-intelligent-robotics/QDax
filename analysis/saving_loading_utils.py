import datetime
import os
import os.path as osp
import pickle
import uuid
from typing import Any

from qdax.training.configuration import Configuration


def save_dataclass(
    dataclass_object,
    folder: str,
    name_file: str,
) -> None:
  path_save_file = os.path.join(folder, name_file)
  with open(path_save_file, "wb") as file_to_save:
    pickle.dump(dataclass_object, file_to_save)


def load_dataclass(
    path_file_to_load: str,
) -> Any:
  with open(path_file_to_load, "rb") as file_to_load:
    return pickle.load(file_to_load)


# Saving metrics and timings
def make_results_folder(result_path,
                        experiment_name,
                        configuration: Configuration):

  path_folder_replication = osp.join(
    result_path,
    configuration.get_results_folder(experiment_name),
    f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4()}",
  )

  os.makedirs(path_folder_replication)

  return path_folder_replication
