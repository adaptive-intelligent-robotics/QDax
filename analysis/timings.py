import os
import pickle
from dataclasses import dataclass

import analysis.saving_loading_utils as saving_loading_utils


@dataclass
class Timings:
    init_framework: float = 0.0
    init_policies: float = 0.0
    init_QD: float = 0.0
    avg_epoch: float = 0.0
    avg_eval_per_sec: float = 0.0
    full_training: float = 0.0

    def save(
        self,
        folder: str = os.curdir,
        name_file: str = "timings.pkl",
    ):
        saving_loading_utils.save_dataclass(
            dataclass_object=self,
            folder=folder,
            name_file=name_file,
        )
