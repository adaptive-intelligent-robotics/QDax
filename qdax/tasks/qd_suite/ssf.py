import time
from typing import Tuple

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from qdax.tasks.hypervolume_functions import square_scoring_function
from qdax.tasks.qd_suite.qd_suite_task import QDSuiteTask
from qdax.types import Descriptor, Fitness, Genotype


class SsfV0(QDSuiteTask):
    def __init__(
        self,
        param_size: int,
    ):
        self.param_size = param_size

    def evaluation(
        self,
        params: Genotype,
    ) -> Tuple[Fitness, Descriptor]:
        norm = jnp.linalg.norm(params, ord=2)
        r_2k_plus_1, _, k = self._get_k(params)
        index = jnp.floor(norm / r_2k_plus_1)
        bd = jax.lax.cond(index == 0.0, lambda: norm, lambda: r_2k_plus_1)
        constant_fitness = jnp.asarray(1.0)
        bd = jnp.asarray(bd).reshape((self.get_bd_size(),))
        return constant_fitness, bd

    def get_bd_size(self) -> int:
        return 1

    def get_min_max_descriptor(self) -> Tuple[float, float]:
        return 0.0, jnp.inf

    def get_bounded_min_max_descriptor(self) -> Tuple[float, float]:
        return 0.0, 1000.0

    def get_min_max_params(self) -> Tuple[float, float]:
        return -jnp.inf, jnp.inf

    def get_initial_parameters(self, batch_size: int) -> Genotype:
        return jnp.zeros(shape=(batch_size, self.param_size))

    def _get_k(self, params: Genotype) -> Tuple[float, float, int]:
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


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    ssf_v0 = SsfV0(param_size=1).evaluation

    print(ssf_v0(jnp.array([4])))
    x = jnp.linspace(0, 2000, 2048)
    x = jnp.reshape(x, (-1, 1))
    print(jax.vmap(ssf_v0)(x))

    f = square_scoring_function
    print(jax.vmap(ssf_v0)(x))
    start = time.time()
    print(jax.vmap(ssf_v0)(x))
    print("Time elapsed:", time.time() - start)
    plt.plot(x, jax.vmap(ssf_v0)(x)[1])
    plt.show()
