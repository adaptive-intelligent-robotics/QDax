import time
from enum import Enum
from functools import partial
from typing import Tuple

import jax.lax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from qdax.types import Fitness, Descriptor, Genotype


class ParameterizationGenotype(Enum):
    angle = "angle"
    arc_length = "arc_length"


class ArchimedeanBD(Enum):
    euclidean = "euclidean"
    geodesic = "geodesic"


def archimedean_spiral(params: Genotype,
                       parameterization: ParameterizationGenotype,
                       archimedean_bd: ArchimedeanBD,
                       parameter=0.01,
                       precision=None,
                       alpha=40,
                       ) -> Tuple[Fitness, Descriptor]:
    """
    Seach space should be [0,alpha * pi]^n
    BD space should be [0,1]^n
    """
    if precision is None:
        precision = alpha * jnp.pi / 1e7

    constant_fitness = jnp.asarray(1.)

    def _gamma(angle):
        return jnp.asarray([parameter * angle * jnp.cos(angle),
                            parameter * angle * jnp.sin(angle)])

    def _arc_length(angle):
        return (parameter / 2) * \
               (angle * jnp.sqrt(1 + jnp.power(angle,2))
                + jnp.log(angle + jnp.sqrt(1 + jnp.power(angle, 2))))

    def _cond_fun(elem: Tuple[float, float, float]) -> jnp.bool_:
        inf, sup, target = elem
        return (sup - inf) > precision

    def _body_fun(elem: Tuple[float, float, float]) -> Tuple[float, float, float]:
        inf, sup, target_angle_length = elem
        middle = (sup + inf) / 2.
        arc_length_middle = _arc_length(middle)
        new_inf, new_sup = jax.lax.cond(target_angle_length < arc_length_middle,
                                        lambda: (inf, middle),
                                        lambda: (middle, sup))
        return new_inf, new_sup, target_angle_length

    def _approximate_angle_from_arc_length(target_arc_length: float) -> jnp.ndarray:
        inf, sup, _ = jax.lax.while_loop(_cond_fun,
                                         _body_fun,
                                         init_val=(0.,
                                                   alpha * jnp.pi,
                                                   target_arc_length))
        middle = (sup + inf) / 2.
        return jnp.asarray(middle)

    if archimedean_bd == ArchimedeanBD.geodesic and parameterization == ParameterizationGenotype.arc_length:
        arc_length = params
        return constant_fitness, arc_length
    elif archimedean_bd == ArchimedeanBD.geodesic and parameterization == ParameterizationGenotype.angle:
        arc_length = _arc_length(params)
        return constant_fitness, arc_length
    elif archimedean_bd == ArchimedeanBD.euclidean and parameterization == ParameterizationGenotype.arc_length:
        arc_length = params
        angle = _approximate_angle_from_arc_length(arc_length[0])
        euclidean_bd = _gamma(angle)
        return constant_fitness, euclidean_bd
    elif archimedean_bd == ArchimedeanBD.euclidean and parameterization == ParameterizationGenotype.angle:
        return constant_fitness, _gamma(params)




def get_arc_length(angle,
                a
                ):
    return (a / 2) * (angle * jnp.sqrt(1 + jnp.power(angle,
                                                     2)) + jnp.log(angle + jnp.sqrt(1 + jnp.power(angle, 2))))


if __name__ == '__main__':
    parameter = 200
    alpha = 10

    archimedean_spiral_fn = partial(archimedean_spiral,
                                    parameterization=ParameterizationGenotype.arc_length,
                                    archimedean_bd=ArchimedeanBD.euclidean,
                                    parameter=parameter,)

    max_length = get_arc_length(jnp.pi * alpha,
                                parameter)
    x = jnp.linspace(0,
                     max_length,
                     16000).reshape((-1, 1))
    f = jax.jit(jax.vmap(archimedean_spiral_fn))
    f(x)
    start = time.time()
    res = f(x)
    print(res)
    print("time taken: {}".format(time.time() - start))
    plt.plot(res[1][:, 0],
             res[1][:, 1])
    plt.show()
