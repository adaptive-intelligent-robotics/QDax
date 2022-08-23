import abc
from typing import Tuple, Union

import jax
from jax import numpy as jnp

from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class QDSuiteTask(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluation(self, params: Genotype) -> Tuple[Fitness, Descriptor]:
        ...

    def scoring_function(
        self,
        params: Genotype,
        random_key: RNGKey,
    ) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
        """
        Evaluate params in parallel
        """
        fitnesses, descriptors = jax.vmap(self.evaluation)(params)

        return fitnesses, descriptors, {}, random_key

    @abc.abstractmethod
    def get_descriptor_size(self) -> int:
        ...

    @abc.abstractmethod
    def get_min_max_descriptor(
        self,
    ) -> Tuple[Union[float, jnp.ndarray], Union[float, jnp.ndarray]]:
        ...

    def get_bounded_min_max_descriptor(
        self,
    ) -> Tuple[Union[float, jnp.ndarray], Union[float, jnp.ndarray]]:
        min_bd, max_bd = self.get_min_max_descriptor()
        if jnp.isinf(max_bd) or jnp.isinf(min_bd):
            raise NotImplementedError(
                "Boundedness has not been implemented " "for this unbounded task"
            )
        else:
            return min_bd, max_bd

    @abc.abstractmethod
    def get_min_max_params(
        self,
    ) -> Tuple[Union[float, jnp.ndarray], Union[float, jnp.ndarray]]:
        ...

    @abc.abstractmethod
    def get_initial_parameters(self, batch_size: int) -> Genotype:
        ...
