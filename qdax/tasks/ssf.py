import time
from typing import Tuple

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from qdax.tasks.hypervolume_functions import square_scoring_function, \
    get_scoring_function
from qdax.types import Descriptor, Fitness, Genotype


def ssf_v0(params: Genotype) -> Tuple[Fitness, Descriptor]:
    norm = jnp.linalg.norm(params,
                           ord=2)
    R_2k_plus_1, _, k = _get_k(params)
    index = jnp.floor(norm / R_2k_plus_1)
    bd = jax.lax.cond(index == 0.,
                      lambda: norm,
                      lambda: R_2k_plus_1)
    fitness = jnp.asarray(1.)
    return fitness, bd


def _get_k(params: Genotype, ) -> Tuple[float, float, int]:
    norm_params = jnp.linalg.norm(params,
                                  ord=2)
    init_k = 0
    R_0 = 0.
    R_1 = R_0 + _get_R_add_odd(init_k)
    (R_2k, R_2k_plus_1), norm, k = jax.lax.while_loop(_cond_fun,
                                                      _body_fun,
                                                      ((0., R_1),
                                                       norm_params,
                                                       init_k))
    # R_2k_plus_1 = R_2k_plus_2 - _get_R_add_even(k_plus_1)
    # k = k_plus_1 - 1
    return R_2k_plus_1, norm, k


def _cond_fun(elem: Tuple[Tuple[float, float], float, int]) -> jnp.bool_:
    (R_2k, R_2k_plus_1), norm, k = elem
    return R_2k_plus_1 + _get_R_add_even(k + 1) < norm


def _body_fun(elem: Tuple[Tuple[float, float], float, int]) -> Tuple[
    Tuple[float, float], float, int]:
    (R_2k, R_2k_plus_1), norm, k = elem
    k_plus_1 = k + 1
    R_2k_plus_2 = R_2k_plus_1 + _get_R_add_even(k_plus_1)
    R_2k_plus_3 = R_2k_plus_2 + _get_R_add_odd(k_plus_1)
    return (R_2k_plus_2, R_2k_plus_3), norm, k + 1


def _get_R_add_odd(k: int) -> float:
    return 2 * (k ** 3) + 1


def _get_R_add_even(k: int) -> float:
    return 2 * ((k - 1) ** 3) + 1


ssf_v0_scoring_function = get_scoring_function(task_fn=ssf_v0)

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    print(ssf_v0(jnp.array([4])))
    x = jnp.linspace(0,
                     2000,
                     2048)
    x = jnp.reshape(x,
                    (-1, 1))
    print(jax.vmap(ssf_v0)(x))

    f = square_scoring_function
    print(jax.vmap(ssf_v0)(x))
    start = time.time()
    print(jax.vmap(ssf_v0)(x))
    print("Time elapsed:",
          time.time() - start)
    plt.plot(x,
             jax.vmap(ssf_v0)(x)[1])
    plt.show()
