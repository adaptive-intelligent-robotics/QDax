import os
import pickle
from typing import Any
import jax.numpy as jnp

from dataclasses import dataclass

import qdax.stats.saving_loading_utils as saving_loading_utils
Array = Any

#@dataclass
class Timings:
  def __init__(self, log_frequency, num_epochs):
    self.init_framework: float = 0.0
    self.init_env: float = 0.0
    self.init_policies: float = 0.0
    self.init_QD: float = 0.0
    self.avg_epoch: float = 0.0
    self.avg_eval_per_sec: float = 0.0
    self.full_training: float = 0.0

    log_size = jnp.ceil(num_epochs / log_frequency).astype(int) + 1
    self.epoch_runtime: Array = jnp.zeros([log_size, 1])

  def save(self,
           folder: str = os.curdir,
           name_file: str = "timings.pkl",
           ):
    saving_loading_utils.save_dataclass(
      dataclass_object=self,
      folder=folder,
      name_file=name_file,
    )
