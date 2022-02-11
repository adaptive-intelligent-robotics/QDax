import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Configuration:
    implementation: str
    env_name: str
    grid_shape: tuple
    genotype_dim: int
    num_epochs: int
    num_evals: int
    population_size: int
    dump_period: int
    min_bd: float  # Assume  each BD dimension has same bounds
    max_bd: float
    qd_params: Dict[str, Any]

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
        current_results_dict.pop("dump_period")
        current_results_dict.pop("min_bd")
        current_results_dict.pop("max_bd")
        current_results_dict.pop("qd_params")

        variables_to_avoid = [
            "parallel",
            "cvt_use_cache",
            "dump_period",
            "random_init_batch",
            "min",
            "max",
            "cvt_samples",
            "random_init",
            "iso_sigma",
            "line_sigma"
            "batch_size",
        ]

        return f"{experiment_name}{self.fix_name_folder(self.get_all_variables_str_from_dict(current_results_dict, variables_to_avoid=variables_to_avoid))}"

    @staticmethod
    def get_all_variables_str_from_dict(dictionary, variables_to_avoid=None):
        if variables_to_avoid is None:
            variables_to_avoid = []

        all_variables_str = ""

        for variable_str, value in dictionary.items():
            if variable_str in variables_to_avoid:
                continue

            print("Variable str: ", variable_str, "value: ", value)
            if isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
                all_variables_str = all_variables_str + f"_{variable_str}-{value}"
            elif isinstance(value, dict):
                all_variables_str += Configuration.get_all_variables_str_from_dict(value, variables_to_avoid)
            elif isinstance(value, tuple):
                all_variables_str = all_variables_str + f"_{variable_str}-{'-'.join(map(str, value))}"
            else:
                print(
                    f"WARNING: Unexpected type encountered when computing results folder name, for variable {variable_str}")

        return Configuration.fix_name_folder(all_variables_str)

    @staticmethod
    def fix_name_folder(name_folder):
        return name_folder \
            .strip() \
            .lower() \
            .replace(' ', '') \
            .replace('.', '-') \
            .replace('=', '-')
