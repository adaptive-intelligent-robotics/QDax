import os
import pickle

import flax.struct
from brax import envs
from brax.training.types import PRNGKey

import qdax.stats.saving_loading_utils as saving_loading_utils
from qdax.qd_utils import grid_archive
from qdax.stats.metrics import Metrics


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    key: PRNGKey
    repertoire: grid_archive.Repertoire
    metrics: Metrics
    state: envs.State

    def save(
        self,
        folder: str = os.curdir,
        name_file: str = "training_state.pkl",
    ) -> None:
        saving_loading_utils.save_dataclass(
            dataclass_object=self,
            folder=folder,
            name_file=name_file,
        )
