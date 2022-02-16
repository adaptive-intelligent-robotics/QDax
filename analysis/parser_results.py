import os
from typing import Dict, List

import argparse
from qdax.training.configuration import Configuration

CONFIGURATION_FILE_NAME = "configuration.json"
TRAINING_STATE_PKL_FILE = "training_state.pkl"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, help="where to load the results from")
    return parser.parse_args()


def get_all_folders_results(path_results_folder):
    return [
        path_folder
        for path_folder in os.listdir(path_results_folder)
        if path_folder.startswith("qdax_training_")
    ]


def _get_configuration_one_folder(folder_result):
    return Configuration.load_from_json(
        os.path.join(folder_result, CONFIGURATION_FILE_NAME)
    )


def get_configurations_per_folder(all_folders_results):
    return {
        folder_result: _get_configuration_one_folder(folder_result)
        for folder_result in all_folders_results
    }


def _search_existing_configuration(all_configurations_per_folder, groups_same_configuration, configuration):
    for folder in groups_same_configuration:
        other_configuration = all_configurations_per_folder[folder]
        if other_configuration == configuration:
            return folder
    return None


def get_groups_same_configurations(configurations_per_folder: Dict[str, Configuration]) -> Dict[str, List[str]]:
    groups_same_configuration = dict()

    for folder_result_path, configuration in configurations_per_folder.items():

        folder_configuration = _search_existing_configuration(configurations_per_folder,
                                                              groups_same_configuration,
                                                              configuration)

        if folder_configuration is not None:
            groups_same_configuration[folder_configuration].append(folder_configuration)
        else:
            groups_same_configuration[folder_result_path] = [folder_result_path]

    return groups_same_configuration


def main():
    args = get_args()
    path_results_folder = args.results
    all_folders_results = get_all_folders_results(path_results_folder)
    configurations_per_folder = get_configurations_per_folder(all_folders_results)
    groups_same_configuration = get_groups_same_configurations(configurations_per_folder)
    print(groups_same_configuration)


if __name__ == '__main__':
    main()
