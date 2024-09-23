import abc
from typing import Tuple, Union

import jax
from jax import numpy as jnp

from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class QDSuiteTask(abc.ABC):
    @abc.abstractmethod
    def evaluation(self, params: Genotype) -> Tuple[Fitness, Descriptor]:
        """
        The function evaluation computes the fitness and the descriptor of the
        parameters passed as input.

        Args:
            params: The batch of parameters to evaluate

        Returns:
            The fitnesses and the descriptors of the parameters.
        """
        ...

    def scoring_function(
        self,
        params: Genotype,
        key: RNGKey,
    ) -> Tuple[Fitness, Descriptor, ExtraScores]:
        """
        Evaluate params in parallel
        """
        fitnesses, descriptors = jax.vmap(self.evaluation)(params)

        return fitnesses, descriptors, {}

    @abc.abstractmethod
    def get_descriptor_size(self) -> int:
        """
        The function get_descriptor_size returns the size of the descriptor.

        Returns:
            The size of the descriptor.
        """
        ...

    @abc.abstractmethod
    def get_min_max_descriptor(
        self,
    ) -> Tuple[Union[float, jnp.ndarray], Union[float, jnp.ndarray]]:
        """
        Get the minimum and maximum descriptor values.

        Returns:
            The minimum and maximum descriptor values.
        """
        ...

    def get_bounded_min_max_descriptor(
        self,
    ) -> Tuple[Union[float, jnp.ndarray], Union[float, jnp.ndarray]]:
        """
        Returns:
            The minimum and maximum descriptor assuming that
            the descriptor space is bounded.
        """
        min_descriptor, max_descriptor = self.get_min_max_descriptor()
        if jnp.isinf(max_descriptor) or jnp.isinf(min_descriptor):
            raise NotImplementedError(
                "Boundedness has not been implemented " "for this unbounded task"
            )
        else:
            return min_descriptor, max_descriptor

    @abc.abstractmethod
    def get_min_max_params(
        self,
    ) -> Tuple[Union[float, jnp.ndarray], Union[float, jnp.ndarray]]:
        """
        Get the minimum and maximum parameter values.

        Returns:
            The minimum and maximum parameter values.
        """
        ...

    @abc.abstractmethod
    def get_initial_parameters(self, batch_size: int) -> Genotype:
        """
        Get the initial parameters.

        Args:
            batch_size: The batch size.

        Returns:
            The initial parameters.
        """
        ...
