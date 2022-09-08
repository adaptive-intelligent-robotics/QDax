from typing import Tuple

import jax
import jax.numpy as jnp

from qdax.tasks.qd_suite.qd_suite_task import QDSuiteTask
from qdax.types import Descriptor, Fitness, Genotype


class SsfV0(QDSuiteTask):
    def __init__(
        self,
        param_size: int,
    ):
        """
        Implements the Self-Similar Function (SSF) task
        from Achkan Salehi and Stephane Doncieux.

        Args:
            param_size: The number of parameters in the genotype.
        """
        self.param_size = param_size

    def evaluation(
        self,
        params: Genotype,
    ) -> Tuple[Fitness, Descriptor]:
        """
        The function evaluation computes the fitness and the descriptor of the
        parameters passed as input. The fitness is always 1.0 as the task does
        not consider elitism.

        Args:
            params: The batch of parameters to evaluate

        Returns:
            The fitnesses and the descriptors of the parameters.
        """
        norm = jnp.linalg.norm(params, ord=2)
        r_2k_plus_1, _, k = self._get_k(params)
        index = jnp.floor(norm / r_2k_plus_1)
        bd = jax.lax.cond(index == 0.0, lambda: norm, lambda: r_2k_plus_1)
        constant_fitness = jnp.asarray(1.0)
        bd = jnp.asarray(bd).reshape((self.get_descriptor_size(),))
        return constant_fitness, bd

    def get_descriptor_size(self) -> int:
        """
        Returns:
            The descriptor size.
        """
        return 1

    def get_min_max_descriptor(self) -> Tuple[float, float]:
        """
        Returns:
            The minimum and maximum descriptor.
        """
        return 0.0, jnp.inf

    def get_bounded_min_max_descriptor(self) -> Tuple[float, float]:
        """
        Returns:
            The minimum and maximum descriptor assuming that
            the descriptor space is bounded.
        """
        return 0.0, 1000.0

    def get_min_max_params(self) -> Tuple[float, float]:
        """
        Returns:
            The minimum and maximum parameters (here
            the parameter space is unbounded).
        """
        return -jnp.inf, jnp.inf

    def get_initial_parameters(self, batch_size: int) -> Genotype:
        """
        Returns:
            The initial parameters (of size batch_size x param_size).
        """
        return jnp.zeros(shape=(batch_size, self.param_size))

    def _get_k(self, params: Genotype) -> Tuple[float, float, int]:
        """
        Computes the k-th level of the SSF.

        Args:
            params: The parameters to evaluate.

        Returns:
            (R_2k_plus_1, norm of params, k)
        """
        norm_params = jnp.linalg.norm(params, ord=2)
        init_k = 0
        r_0 = 0.0
        r_1 = r_0 + self._get_r_add_odd(init_k)
        (r_2k, r_2k_plus_1), norm, k = jax.lax.while_loop(
            self._cond_fun, self._body_fun, ((0.0, r_1), norm_params, init_k)
        )
        return r_2k_plus_1, norm, k

    def _cond_fun(self, elem: Tuple[Tuple[float, float], float, int]) -> jnp.bool_:
        (r_2k, r_2k_plus_1), norm, k = elem
        return r_2k_plus_1 + self._get_r_add_even(k + 1) < norm

    def _body_fun(
        self, elem: Tuple[Tuple[float, float], float, int]
    ) -> Tuple[Tuple[float, float], float, int]:
        (r_2k, r_2k_plus_1), norm, k = elem
        k_plus_1 = k + 1
        r_2k_plus_2 = r_2k_plus_1 + self._get_r_add_even(k_plus_1)
        r_2k_plus_3 = r_2k_plus_2 + self._get_r_add_odd(k_plus_1)
        return (r_2k_plus_2, r_2k_plus_3), norm, k + 1

    def _get_r_add_odd(self, k: int) -> float:
        return 2 * (k**3) + 1

    def _get_r_add_even(self, k: int) -> float:
        return 2 * ((k - 1) ** 3) + 1
