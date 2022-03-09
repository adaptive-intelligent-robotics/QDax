import abc
import os
from typing import Union

import numpy as np

from analysis import data_reader as data_reader
from analysis import saving_loading_utils
from analysis.configuration import Configuration as ConfigurationQDBaselines
from analysis.timings import Timings as TimingsQDBaselines
from qdax.stats.timings import Timings as TimingsQDAX
from qdax.training.configuration import Configuration as ConfigurationQDAX


class DataLoader(metaclass=abc.ABCMeta):
    EPOCH = "Epoch"
    EVALUATIONS = "Evaluations"
    ARCHIVE_SIZE = "Archive Size"
    BEST_FIT = "Best Fit"
    QD_SCORE = "QD Score"

    TRAINING_STATE_FILE_NAME = "training_state.pkl"
    LOG_FILE_DAT = "log_file.dat"
    CONFIGURATION_FILE_NAME = "configuration.json"
    TIMINGS_PKL_FILE_NAME = "timings.pkl"

    DATALOADER_QDAX_NAME = "qdax"
    DATALOADER_PYMAPELITES_NAME = "pymapelites"
    DATALOADER_PYRIBS_NAME = "pyribs"

    @staticmethod
    def get_dataloader_from_name_dict():
        return {
            DataLoader.DATALOADER_QDAX_NAME: DataLoaderQDAX(),
            DataLoader.DATALOADER_PYMAPELITES_NAME: DataLoaderPyMAPElites(),
            DataLoader.DATALOADER_PYRIBS_NAME: DataLoaderPyRibs(),
        }

    @staticmethod
    def get_configuration_json_path(path_replication):
        return os.path.join(
            path_replication,
            DataLoader.CONFIGURATION_FILE_NAME,
        )

    @staticmethod
    def get_timings_pkl_path(path_replication):
        return os.path.join(
            path_replication,
            DataLoader.TIMINGS_PKL_FILE_NAME,
        )

    @classmethod
    @abc.abstractmethod
    def load_json(cls, path_json):
        ...

    @classmethod
    @abc.abstractmethod
    def get_scores_replication(cls, path_replication):
        ...

    @classmethod
    @abc.abstractmethod
    def get_dict_scores(cls, path_replication):
        ...

    @classmethod
    def get_timings(cls, path_replication) -> Union[TimingsQDAX, TimingsQDBaselines]:
        timings_pkl_path = cls.get_timings_pkl_path(path_replication)
        timings_obj = saving_loading_utils.load_dataclass(
            path_file_to_load=timings_pkl_path
        )
        return timings_obj


class DataLoaderQDAX(DataLoader):
    @classmethod
    def load_json(cls, path_json) -> ConfigurationQDAX:
        return ConfigurationQDAX.load_from_json(path_json)

    @classmethod
    def get_scores_replication(cls, path_replication):
        path_training_state = os.path.join(
            path_replication,
            cls.TRAINING_STATE_FILE_NAME,
        )

        training_data = data_reader.load_training_state(
            path_training_state
        )  # type: TrainingState
        scores = training_data.metrics.scores

        return scores

    @classmethod
    def get_dict_scores(cls, path_replication):
        scores_array = cls.get_scores_replication(path_replication)
        configuration_json_path = cls.get_configuration_json_path(path_replication)
        batch_size = cls.load_json(configuration_json_path).population_size

        return {
            cls.EPOCH: scores_array[:, 0],
            cls.EVALUATIONS: scores_array[:, 0] * batch_size,
            cls.ARCHIVE_SIZE: scores_array[:, 1],
            cls.BEST_FIT: scores_array[:, 2],
            cls.QD_SCORE: scores_array[:, 3],
        }


class DataLoaderPyMAPElites(DataLoader):
    @classmethod
    def load_json(cls, path_json) -> ConfigurationQDBaselines:
        return ConfigurationQDBaselines.load_from_json(path_json)

    @classmethod
    def get_scores_replication(cls, path_replication):
        path_log_file = os.path.join(
            path_replication,
            cls.LOG_FILE_DAT,
        )

        all_logs_array = data_reader.read_data(
            file_path=path_log_file,
            list_names_components=["_"],
        )["_"]

        index_epoch = 0
        index_archive_size = 1
        index_best_fit = 4
        index_qd_score = 3

        scores_array = all_logs_array[
            :, (index_epoch, index_archive_size, index_best_fit, index_qd_score)
        ]

        return scores_array

    @classmethod
    def get_dict_scores(cls, path_replication):
        scores_array = cls.get_scores_replication(path_replication)
        configuration_json_path = cls.get_configuration_json_path(path_replication)
        batch_size = cls.load_json(configuration_json_path).population_size

        return {
            cls.EPOCH: scores_array[:, 0],
            cls.EVALUATIONS: scores_array[:, 0] * batch_size,
            cls.ARCHIVE_SIZE: scores_array[:, 1],
            cls.BEST_FIT: scores_array[:, 2],
            cls.QD_SCORE: scores_array[:, 3],
        }


class DataLoaderPyRibs(DataLoader):
    @classmethod
    def load_json(cls, path_json) -> ConfigurationQDBaselines:
        return ConfigurationQDBaselines.load_from_json(path_json)

    @classmethod
    def get_scores_replication(cls, path_replication):
        path_log_file = os.path.join(
            path_replication,
            cls.LOG_FILE_DAT,
        )

        all_logs_array = data_reader.read_data(
            file_path=path_log_file,
            list_names_components=["_"],
        )["_"]

        index_epoch = 0
        index_archive_size = 1
        index_best_fit = 4
        index_qd_score = 3

        scores_array = all_logs_array[
            :, (index_epoch, index_archive_size, index_best_fit, index_qd_score)
        ]

        return scores_array

    @classmethod
    def get_dict_scores(cls, path_replication):
        scores_array = cls.get_scores_replication(path_replication)
        configuration_json_path = cls.get_configuration_json_path(path_replication)
        batch_size = cls.load_json(configuration_json_path).population_size

        return {
            cls.EPOCH: scores_array[:, 0],
            cls.EVALUATIONS: scores_array[:, 0] * batch_size,
            cls.ARCHIVE_SIZE: scores_array[:, 1],
            cls.BEST_FIT: scores_array[:, 2],
            cls.QD_SCORE: scores_array[:, 3],
        }
