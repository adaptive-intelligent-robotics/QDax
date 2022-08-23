import time
from enum import Enum
from typing import Optional, Tuple, Union

import jax.lax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from qdax.tasks.qd_suite.qd_suite_task import QDSuiteTask
from qdax.types import Descriptor, Fitness, Genotype


class ParameterizationGenotype(Enum):
    angle = "angle"
    arc_length = "arc_length"


class ArchimedeanBD(Enum):
    euclidean = "euclidean"
    geodesic = "geodesic"


class ArchimedeanSpiralV0(QDSuiteTask):
    def __init__(
        self,
        parameterization: ParameterizationGenotype,
        archimedean_bd: ArchimedeanBD,
        parameter: float = 0.01,
        precision: Optional[float] = None,
        alpha: float = 40.0,
    ):
        self.parameterization = parameterization
        self.archimedean_bd = archimedean_bd
        self.parameter = parameter
        if precision is None:
            self.precision = alpha * jnp.pi / 1e7
        else:
            self.precision = precision
        self.alpha = alpha

    def _gamma(self, angle: float) -> jnp.ndarray:
        return jnp.hstack(
            [
                self.parameter * angle * jnp.cos(angle),
                self.parameter * angle * jnp.sin(angle),
            ]
        )

    def get_arc_length(self, angle: Union[float, jnp.ndarray]) -> jnp.ndarray:
        return (self.parameter / 2) * (
            angle * jnp.sqrt(1 + jnp.power(angle, 2))
            + jnp.log(angle + jnp.sqrt(1 + jnp.power(angle, 2)))
        )

    def _cond_fun(self, elem: Tuple[float, float, float]) -> jnp.bool_:
        inf, sup, target = elem
        return (sup - inf) > self.precision

    def _body_fun(self, elem: Tuple[float, float, float]) -> Tuple[float, float, float]:
        inf, sup, target_angle_length = elem
        middle = (sup + inf) / 2.0
        arc_length_middle = self.get_arc_length(middle)
        new_inf, new_sup = jax.lax.cond(
            target_angle_length < arc_length_middle,
            lambda: (inf, middle),
            lambda: (middle, sup),
        )
        return new_inf, new_sup, target_angle_length

    def _approximate_angle_from_arc_length(
        self, target_arc_length: float
    ) -> jnp.ndarray:
        inf, sup, _ = jax.lax.while_loop(
            self._cond_fun,
            self._body_fun,
            init_val=(0.0, self.alpha * jnp.pi, target_arc_length),
        )
        middle = (sup + inf) / 2.0
        return jnp.asarray(middle)

    def evaluation(self, params: Genotype) -> Tuple[Fitness, Descriptor]:
        constant_fitness = jnp.asarray(1.0)

        if (
            self.archimedean_bd == ArchimedeanBD.geodesic
            and self.parameterization == ParameterizationGenotype.arc_length
        ):
            arc_length = params
            return constant_fitness, arc_length
        elif (
            self.archimedean_bd == ArchimedeanBD.geodesic
            and self.parameterization == ParameterizationGenotype.angle
        ):
            angle = params
            arc_length = self.get_arc_length(angle)
            return constant_fitness, arc_length
        elif (
            self.archimedean_bd == ArchimedeanBD.euclidean
            and self.parameterization == ParameterizationGenotype.arc_length
        ):
            arc_length = params
            angle = self._approximate_angle_from_arc_length(arc_length[0])
            euclidean_bd = self._gamma(angle)
            return constant_fitness, euclidean_bd
        elif (
            self.archimedean_bd == ArchimedeanBD.euclidean
            and self.parameterization == ParameterizationGenotype.angle
        ):
            angle = params
            return constant_fitness, self._gamma(angle)
        else:
            raise ValueError("Invalid parameterization and/or BD")

    def get_bd_size(self) -> int:
        if self.archimedean_bd == ArchimedeanBD.euclidean:
            return 2
        elif self.archimedean_bd == ArchimedeanBD.geodesic:
            return 1
        else:
            raise ValueError("Invalid BD")

    def get_min_max_descriptor(self) -> Tuple[float, float]:
        max_angle = self.alpha * jnp.pi
        max_norm = jnp.linalg.norm(self._gamma(max_angle))

        if self.archimedean_bd == ArchimedeanBD.euclidean:
            return -max_norm, max_norm
        elif self.archimedean_bd == ArchimedeanBD.geodesic:
            max_arc_length = self.get_arc_length(max_angle)
            return 0.0, max_arc_length
        else:
            raise ValueError("Invalid BD")

    def get_min_max_params(self) -> Tuple[float, float]:
        if self.parameterization == ParameterizationGenotype.angle:
            max_angle = self.alpha * jnp.pi
            return 0.0, max_angle
        elif self.parameterization == ParameterizationGenotype.arc_length:
            max_angle = self.alpha * jnp.pi
            max_arc_length = self.get_arc_length(max_angle)
            return 0, max_arc_length
        else:
            raise ValueError("Invalid parameterization")

    def get_initial_parameters(self, batch_size: int) -> Genotype:
        max_angle = self.alpha * jnp.pi
        mid_angle = max_angle / 2.0
        mid_number_turns = 1 + int(mid_angle / (2.0 * jnp.pi))
        horizontal_left_mid_angle = mid_number_turns * jnp.pi * 2

        if self.parameterization == ParameterizationGenotype.angle:
            angle_array = jnp.asarray(horizontal_left_mid_angle).reshape((1, 1))
            return jnp.repeat(angle_array, batch_size, axis=0)
        elif self.parameterization == ParameterizationGenotype.arc_length:
            arc_length = self.get_arc_length(horizontal_left_mid_angle)
            length_array = jnp.asarray(arc_length).reshape((1, 1))
            return jnp.repeat(length_array, batch_size, axis=0)
        else:
            raise ValueError("Invalid parameterization")


if __name__ == "__main__":
    task = ArchimedeanSpiralV0(
        parameterization=ParameterizationGenotype.arc_length,
        archimedean_bd=ArchimedeanBD.euclidean,
    )
    archimedean_spiral_fn = task.evaluation

    # max_length = task.get_arc_length(jnp.pi * task.alpha)
    x = jnp.linspace(*task.get_min_max_params(), num=16000).reshape((-1, 1))
    f = jax.jit(jax.vmap(archimedean_spiral_fn))
    f(x)
    start = time.time()
    res = f(x)

    point = task.get_initial_parameters(126)
    red_point = f(point)

    print(res)
    print("time taken: {}".format(time.time() - start))
    plt.plot(res[1][:, 0], res[1][:, 1])
    plt.scatter(red_point[1][0, 0], red_point[1][0, 1], c="r")
    plt.show()
