import json
import os
from typing import Dict, Any, Optional, Tuple

import dataclasses
from dataclasses import dataclass


@dataclass
class Configuration:
  env_name: str
  num_epochs: int
  episode_length: int
  action_repeat: int
  population_size: int
  seed: int
  log_frequency: int
  qd_params: Dict[str, Any]
  min_bd: float  # Assume  each BD dimension has same bounds
  max_bd: float
  grid_shape: tuple
  max_devices_per_host: Optional[int] = None

  def save_to_json(self,
                   folder: str = os.curdir,
                   name_file: str = "configuration.json",
                   ):
    path_file = os.path.join(folder, name_file)
    with open(path_file, "w") as json_file:
      json.dump(dataclasses.asdict(self),
                json_file,
                indent=1,
                )

  @classmethod
  def load_from_json(cls,
                     path_file: str
                     ) -> 'Configuration':
    with open(path_file, "r") as json_file:
      configuration_dictionary = json.load(json_file)

    return cls(**configuration_dictionary)

  def get_results_folder(self, experiment_name):
    current_results_dict = dataclasses.asdict(self)
    current_results_dict.pop("seed")
    current_results_dict.pop("log_frequency")
    current_results_dict.pop("max_devices_per_host")
    current_results_dict.pop("min_bd")
    current_results_dict.pop("max_bd")
    current_results_dict.pop("action_repeat")

    return f"{experiment_name}{self.fix_name_folder(self.get_all_variables_str_from_dict(current_results_dict))}"

  @staticmethod
  def get_all_variables_str_from_dict(dictionary):
    all_variables_str = ""

    for variable_str, value in dictionary.items():
      print("Variable str: ",variable_str, "value: ", value)
      if isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
        all_variables_str = all_variables_str + f"_{variable_str}-{value}"
      elif isinstance(value, dict):
        all_variables_str += Configuration.get_all_variables_str_from_dict(value)
      elif isinstance(value, tuple):
        all_variables_str = all_variables_str + f"_{variable_str}-{'-'.join(map(str, value))}"
      else:
        print(f"WARNING: Unexpected type encountered when computing results folder name, for variable {variable_str}")

    return Configuration.fix_name_folder(all_variables_str)

  @staticmethod
  def fix_name_folder(name_folder):
    return name_folder\
      .strip()\
      .lower()\
      .replace(' ', '')\
      .replace('.', '-')\
      .replace('=', '-')
