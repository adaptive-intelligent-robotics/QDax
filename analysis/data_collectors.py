import glob
import itertools
import json
import operator
import os
from typing import Any, Callable, Dict, TypeVar, ValuesView

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toolz as toolz

import analysis.data_loaders as data_loaders
from analysis import saving_loading_utils


class DataCollectionAllVariants(object):
    def __init__(
        self,
        data_loader: data_loaders.DataLoader,
        main_result_folder: str,
        experiment_name: str,
    ):
        self.data_loader = data_loader
        self.collection_from_variant_str_dict = {
            os.path.basename(variant_path): DataCollectionOneVariant(
                data_loader, variant_path
            )
            for variant_path in self.get_all_variants_folders(
                main_result_folder, experiment_name
            )
        }  # type: Dict[str, DataCollectionOneVariant]

    def __len__(self):
        return len(self.collection_from_variant_str_dict)

    @classmethod
    def get_all_variants_folders(cls, main_result_folder, experiment_name):
        return glob.glob(
            os.path.join(
                main_result_folder,
                f"{experiment_name}_*",
            )
        )

    TMapReturn = TypeVar("TMapReturn")
    TReduceReturn = TypeVar("TReduceReturn")

    def map_and_reduce(
        self,
        map_fn: Callable[["DataCollectionOneReplication"], TMapReturn],
        reduce_fn: Callable[[ValuesView[TMapReturn]], TReduceReturn],
    ) -> Dict[str, TReduceReturn]:
        return {
            variant_folder: reduce_fn(
                data_variant.map_function_to_replication_collection(map_fn).values()
            )
            for variant_folder, data_variant in self.collection_from_variant_str_dict.items()
        }

    def map_function_to_replication_collection(
        self,
        map_fn: Callable[["DataCollectionOneReplication"], TMapReturn],
    ) -> Dict[str, Dict[str, TMapReturn]]:
        return {
            variant_folder: data_variant.map_function_to_replication_collection(map_fn)
            for variant_folder, data_variant in self.collection_from_variant_str_dict.items()
        }

    def apply_function_to_variant_scores(
        self, fn: Callable[[np.ndarray], np.ndarray], name_score
    ):
        return {
            variant_folder: data_variant.apply_function_to_array_scores(fn, name_score)
            for variant_folder, data_variant in self.collection_from_variant_str_dict.items()
        }

    def get_quantile_scores(self, quantile: float, name_score: str):
        return self.apply_function_to_variant_scores(
            fn=lambda array: np.quantile(array, q=quantile, axis=0),
            name_score=name_score,
        )

    def get_median_scores(self, name_score: str):
        return self.get_quantile_scores(
            quantile=0.5,
            name_score=name_score,
        )

    def save_serialised(
        self,
        folder: str = os.curdir,
        name_file: str = "data_collection_all_variants.pkl",
    ):
        saving_loading_utils.save_dataclass(
            dataclass_object=self,
            folder=folder,
            name_file=name_file,
        )

    @staticmethod
    def all_equal(iterable):
        """
        returns true if and only if all the elements in the iterable are equal
        """
        g = itertools.groupby(iterable)
        return next(g, True) and not next(g, False)

    def get_common_attribute(self, get_attribute_fn):
        attribute_per_replication_per_variant = self.get_common_attribute_per_variant(
            get_attribute_fn=get_attribute_fn
        )

        if not DataCollectionAllVariants.all_equal(
            attribute_per_replication_per_variant.values()
        ):
            raise AssertionError(
                f"The requested attribute is not unique among the different variants"
            )

        # Returning one arbitrary element per variant (they are all equal anyway)
        return next(iter(attribute_per_replication_per_variant.values()))

    def get_common_attribute_per_variant(self, get_attribute_fn):
        attribute_per_replication_per_variant = (
            self.map_function_to_replication_collection(map_fn=get_attribute_fn)
        )

        for attribute_per_replication in attribute_per_replication_per_variant.values():
            if not DataCollectionAllVariants.all_equal(
                attribute_per_replication.values()
            ):
                raise AssertionError(
                    f"The requested attribute is not unique among some replications"
                )

        return self.map_and_reduce(
            map_fn=get_attribute_fn,
            reduce_fn=lambda x: next(
                iter(x)
            ),  # getting first attribute (they are all equal anyway)
        )


class DataCollectionOneVariant(object):
    def __init__(
        self,
        data_loader: data_loaders.DataLoader,
        path_one_variant_folder: str,
    ):

        self.collection_from_replication_str_dict = {
            os.path.basename(
                path_replication_one_variant
            ): DataCollectionOneReplication(data_loader, path_replication_one_variant)
            for path_replication_one_variant in self.get_all_paths_replications_one_variant(
                path_one_variant_folder
            )
        }  # type: Dict[str, DataCollectionOneReplication]

    @classmethod
    def get_all_paths_replications_one_variant(cls, path_one_variant_folder: str):
        all_subdirs_variant = os.listdir(path_one_variant_folder)

        all_paths_replications_one_variant = []
        for subdir_variant in all_subdirs_variant:
            path_subdir_variant = os.path.join(path_one_variant_folder, subdir_variant)
            if os.path.exists(
                os.path.join(
                    path_subdir_variant, data_loaders.DataLoader.CONFIGURATION_FILE_NAME
                )
            ):
                all_paths_replications_one_variant.append(path_subdir_variant)

        return all_paths_replications_one_variant

    def get_array_scores(self, name_score: str):
        scores_list = [
            data_replication.metrics_df[name_score].to_numpy()
            for data_replication in self.collection_from_replication_str_dict.values()
        ]

        return np.stack(scores_list, axis=0)

    def apply_function_to_array_scores(self, fn, name_score: str):
        return fn(self.get_array_scores(name_score))

    def map_function_to_replication_collection(
        self, map_fn: Callable[["DataCollectionOneReplication"], Any]
    ):
        return {
            name_replication: map_fn(data_replication)
            for name_replication, data_replication in self.collection_from_replication_str_dict.items()
        }


class DataCollectionOneReplication(object):
    def __init__(
        self, data_loader: data_loaders.DataLoader, path_replication_results: str
    ):
        self.metrics_df = pd.DataFrame(
            data_loader.get_dict_scores(path_replication_results),
        )

        self.configuration = data_loader.load_json(
            data_loader.get_configuration_json_path(
                path_replication=path_replication_results,
            )
        )

        self.timings = data_loader.get_timings(
            path_replication=path_replication_results,
        )


def test():
    # Warning: ensure to always use the adapted DataLoader!
    data_collection_all_variants = DataCollectionAllVariants(
        data_loader=data_loaders.DataLoaderPyMAPElites(),
        main_result_folder="/Users/looka/git/qd_baselines/logs",
        experiment_name="pymapelites",
    )
    # data_collection_all_variants = saving_loading_utils.load_dataclass(
    #     "/Users/looka/git/qd_baselines/analysis/data_collection_all_variants.pkl")

    # Calculating stats of data_collection
    median_qd_scores = data_collection_all_variants.get_median_scores(
        name_score=data_loaders.DataLoader.QD_SCORE
    )

    first_quartile = data_collection_all_variants.get_quantile_scores(
        quantile=0.25, name_score=data_loaders.DataLoader.QD_SCORE
    )
    third_quartile = data_collection_all_variants.get_quantile_scores(
        quantile=0.75, name_score=data_loaders.DataLoader.QD_SCORE
    )

    # Plotting data_collection

    # epochs = data_collection_all_variants.get_median_scores(data_loaders.EPOCH)
    evaluations = data_collection_all_variants.get_median_scores(
        name_score=data_loaders.DataLoader.EVALUATIONS
    )

    for name_variant in median_qd_scores.keys():
        plt.plot(evaluations[name_variant], median_qd_scores[name_variant])
        plt.fill_between(
            evaluations[name_variant],
            first_quartile[name_variant],
            third_quartile[name_variant],
            alpha=0.3,
        )

    plt.ylabel(data_loaders.DataLoader.QD_SCORE)
    plt.xlabel(data_loaders.DataLoader.EVALUATIONS)

    plt.show()

    # Saving the data collections
    data_collection_all_variants.save_serialised(
        name_file="data_collection_all_variants_pymapelites.pkl"
    )

    # Getting all avg_eval_per_sec for example,
    avg_eval_per_sec_dict = (
        data_collection_all_variants.map_function_to_replication_collection(
            map_fn=operator.attrgetter("timings.avg_eval_per_sec")
        )
    )
    print(
        "avg_eval_per_sec per replication", json.dumps(avg_eval_per_sec_dict, indent=4)
    )

    # Gettings medians of avg_eval_per_sec per variant:
    medians_avg_eval_per_sec_dict = data_collection_all_variants.map_and_reduce(
        map_fn=operator.attrgetter("timings.avg_eval_per_sec"),
        reduce_fn=lambda x: np.median(np.fromiter(x, dtype=float), axis=0),
    )
    print(
        "median avg_eval_per_sec per variant",
        json.dumps(medians_avg_eval_per_sec_dict, indent=4),
    )

    # The function map_and_reduce can be used to perform almost any operation we like on the data,
    # For instance we can capture the medians of qd_scores using it:

    median_final_qd_scores = data_collection_all_variants.map_and_reduce(
        map_fn=toolz.compose(
            lambda array_np: array_np.flatten()[
                -1
            ],  # 3. Flattening previously obtained array and getting last value
            lambda df: df[data_loaders.DataLoader.QD_SCORE].to_numpy(),
            # 2. Getting QD_Score and converting it to a numpy array
            operator.attrgetter(
                "metrics_df"
            ),  # 1. Getting metrics_df attribute of DataCollectionOneReplication
        ),
        reduce_fn=toolz.compose(
            lambda x: np.median(
                x, axis=0
            ),  # 5. Getting median value of that list of QD scores
            list,  # 4. For each variant, Converting all values of replications into a list (of final QD scores)
        ),
    )

    print(
        "common attribute per variant (grid_shape)",
        data_collection_all_variants.get_common_attribute_per_variant(
            lambda x: x.configuration.grid_shape
        ),
    )
    print(
        "common attribute (grid_shape)",
        data_collection_all_variants.get_common_attribute(
            lambda x: x.configuration.grid_shape
        ),
    )


if __name__ == "__main__":
    test()
