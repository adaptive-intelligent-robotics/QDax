from enum import Enum
from typing import Optional, Tuple, Union

import jax.lax
import jax.numpy as jnp

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
        amplitude: float = 0.01,
        precision: Optional[float] = None,
        alpha: float = 40.0,
    ):
        """
        Implements the Archimedean spiral task from Salehi et al. (2022):
        https://arxiv.org/abs/2205.03162

        Args:
            parameterization: The parameterization of the genotype,
            can be either angle or arc length.
            archimedean_bd: The Archimedean BD, can be either euclidean or
            geodesic.
            amplitude: The amplitude of the Archimedean spiral.
            precision: The precision of the approximation of the angle from the
            arc length.
            alpha: controls the length/maximal angle of the Archimedean spiral.
        """
        self.parameterization = parameterization
        self.archimedean_bd = archimedean_bd
        self.amplitude = amplitude
        if precision is None:
            self.precision = alpha * jnp.pi / 1e7
        else:
            self.precision = precision
        self.alpha = alpha

    def _gamma(self, angle: Union[float, jnp.ndarray]) -> jnp.ndarray:
        """
        The function gamma is the function that maps the angle to the euclidean
        coordinates of the Archimedean spiral.

        Args:
            angle: The angle of the Archimedean spiral.

        Returns:
            The euclidean coordinates of the Archimedean spiral.
        """
        return jnp.hstack(
            [
                self.amplitude * angle * jnp.cos(angle),
                self.amplitude * angle * jnp.sin(angle),
            ]
        )

    def get_arc_length(self, angle: Union[float, jnp.ndarray]) -> jnp.ndarray:
        """
        The function arc_length is the function that maps the angle to the arc
        length of the Archimedean spiral.

        Args:
            angle: The angle of the Archimedean spiral.

        Returns:
            The arc length of the Archimedean spiral.
        """
        return (self.amplitude / 2) * (
            angle * jnp.sqrt(1 + jnp.power(angle, 2))
            + jnp.log(angle + jnp.sqrt(1 + jnp.power(angle, 2)))
        )

    def _cond_fun(self, elem: Tuple[float, float, float]) -> jnp.bool_:
        """
        The function cond_fun is the function that checks if the precision has
        been reached.

        Args:
            elem: The tuple containing the lower bound, the upper bound and the
                target arc length.

        Returns:
            True if the precision has been reached, False otherwise.
        """
        inf, sup, target = elem
        return (sup - inf) > self.precision

    def _body_fun(self, elem: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        The function body_fun is the function that computes the next iteration
        of the while loop.

        Args:
            elem: The tuple containing the lower bound, the upper bound and the
                target arc length.

        Returns:
            The tuple containing the lower bound, the upper bound and the
            target arc length.
        """
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
        """
        The function approximate_angle_from_arc_length is the function that
        approximates the angle from the arc length.

        Args:
            target_arc_length: The target arc length.

        Returns:
            The angle.
        """
        inf, sup, _ = jax.lax.while_loop(
            self._cond_fun,
            self._body_fun,
            init_val=(0.0, self.alpha * jnp.pi, target_arc_length),
        )
        middle = (sup + inf) / 2.0
        return jnp.asarray(middle)

    def evaluation(self, params: Genotype) -> Tuple[Fitness, Descriptor]:
        """
        The function evaluation computes the fitness and the descriptor of the
        parameters passed as input. The fitness is always 1.0 as no elitism is
        considered.

        Args:
            params: The parameters of the Archimedean spiral.

        Returns:
            The fitness and the descriptor of the parameters.
        """
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

    def get_descriptor_size(self) -> int:
        """
        The function get_descriptor_size returns the size of the descriptor.

        Returns:
            The size of the descriptor.
        """
        if self.archimedean_bd == ArchimedeanBD.euclidean:
            return 2
        elif self.archimedean_bd == ArchimedeanBD.geodesic:
            return 1
        else:
            raise ValueError("Invalid BD")

    def get_min_max_descriptor(self) -> Tuple[float, float]:
        """
        The function get_min_max_descriptor returns the minimum and maximum
        bounds of the descriptor space.

        Returns:
            The minimum and maximum value of the descriptor.
        """
        max_angle = self.alpha * jnp.pi
        max_norm = jnp.linalg.norm(self._gamma(max_angle))

        if self.archimedean_bd == ArchimedeanBD.euclidean:
            return -max_norm, max_norm
        elif self.archimedean_bd == ArchimedeanBD.geodesic:
            max_arc_length = self.get_arc_length(max_angle)
            return 0.0, max_arc_length.item()
        else:
            raise ValueError("Invalid BD")

    def get_min_max_params(self) -> Tuple[float, float]:
        """
        The function get_min_max_params returns the minimum and maximum value
        of the parameter space.

        Returns:
            The minimum and maximum value of the parameters.
        """
        if self.parameterization == ParameterizationGenotype.angle:
            max_angle = self.alpha * jnp.pi
            return 0.0, max_angle
        elif self.parameterization == ParameterizationGenotype.arc_length:
            max_angle = self.alpha * jnp.pi
            max_arc_length = self.get_arc_length(max_angle)
            return 0, max_arc_length.item()
        else:
            raise ValueError("Invalid parameterization")

    def get_initial_parameters(self, batch_size: int) -> Genotype:
        """
        The function get_initial_parameters returns the initial parameters.

        Args:
            batch_size: The batch size.

        Returns:
            The initial parameters (of size batch_size).
        """
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
