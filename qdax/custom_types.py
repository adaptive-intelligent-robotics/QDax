"""Defines some types used in QDax"""

from __future__ import annotations

from typing import Dict, Generic, TypeVar, Union, Any

import flax
import jax
from typing_extensions import TypeAlias

PyTree: TypeAlias = Any

# MDP types
Observation: TypeAlias = jax.Array
Action: TypeAlias = jax.Array
Reward: TypeAlias = jax.Array
Done: TypeAlias = jax.Array
EnvState: TypeAlias = PyTree
Params: TypeAlias = PyTree

# Evolution types
StateDescriptor: TypeAlias = jax.Array
Fitness: TypeAlias = jax.Array
Genotype: TypeAlias = PyTree
Descriptor: TypeAlias = jax.Array
Centroid: TypeAlias = jax.Array
Spread: TypeAlias = jax.Array
Gradient: TypeAlias = jax.Array

Skill: TypeAlias = jax.Array

ExtraScores: TypeAlias = Dict[str, PyTree]

# Pareto fronts
T = TypeVar("T", bound=Union[Fitness, Genotype, Descriptor, jax.Array])


class ParetoFront(Generic[T]):
    def __init__(self) -> None:
        super().__init__()


Mask: TypeAlias = jax.Array

# Others
RNGKey: TypeAlias = jax.Array
Metrics: TypeAlias = Dict[str, jax.Array]


class AuroraExtraInfo(flax.struct.PyTreeNode):
    """
    Information specific to the AURORA algorithm.

    Args:
        model_params: the parameters of the dimensionality reduction model
    """

    model_params: Params


class AuroraExtraInfoNormalization(AuroraExtraInfo):
    """
    Information specific to the AURORA algorithm. In particular, it contains
    the normalization parameters for the observations.

    Args:
        model_params: the parameters of the dimensionality reduction model
        mean_observations: the mean of observations
        std_observations: the std of observations
    """

    mean_observations: jax.Array
    std_observations: jax.Array

    @classmethod
    def create(
        cls,
        model_params: Params,
        mean_observations: jax.Array,
        std_observations: jax.Array,
    ) -> AuroraExtraInfoNormalization:
        return cls(
            model_params=model_params,
            mean_observations=mean_observations,
            std_observations=std_observations,
        )
