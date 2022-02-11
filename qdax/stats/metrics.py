import os
from typing import Any

import flax.struct
import numpy as np
from dataclasses import dataclass
from jax import numpy as jnp

from qdax.stats import saving_loading_utils

Array = Any


@dataclass
class MetricsData:
  archives: np.ndarray
  scores: np.ndarray

  def save(self,
           folder: str = os.curdir,
           name_file: str = "metrics.pkl",
           ):
    saving_loading_utils.save_dataclass(
      dataclass_object=self,
      folder=folder,
      name_file=name_file,
    )

  @classmethod
  def from_metrics(cls,
                   metrics: 'Metrics'
                   ) -> 'MetricsData':
    return cls(
      archives=np.asarray(metrics.archives),
      scores=np.asarray(metrics.scores)
    )


@flax.struct.dataclass
class Metrics:
  archives: Array
  scores: Array

  @classmethod
  def create(cls, num_epochs, log_frequency, grid_shape):
    log_size = jnp.ceil(num_epochs / log_frequency).astype(int) + 1
    archives = jnp.zeros(tuple(jnp.append(jnp.array([log_size]), grid_shape)))
    scores = jnp.zeros([log_size, 4])  # epoch, archive size, best fit, QD score
    return Metrics(archives=archives, scores=scores)

  def save(self,
           folder: str = os.curdir,
           name_file: str = "metrics.pkl",
           ):
    saving_loading_utils.save_dataclass(
      dataclass_object=self,
      folder=folder,
      name_file=name_file,
    )
