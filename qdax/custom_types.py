"""Defines some types used in QDax"""

from __future__ import annotations

from typing import Dict, Generic, TypeVar, Union

import flax
import jax
import jax.numpy as jnp
from chex import ArrayTree
from typing_extensions import TypeAlias

# MDP types
Observation: TypeAlias = jnp.ndarray
Action: TypeAlias = jnp.ndarray
Reward: TypeAlias = jnp.ndarray
Done: TypeAlias = jnp.ndarray
EnvState: TypeAlias = ArrayTree
Params: TypeAlias = ArrayTree

# Evolution types
StateDescriptor: TypeAlias = jnp.ndarray
Fitness: TypeAlias = jnp.ndarray
Genotype: TypeAlias = ArrayTree
Descriptor: TypeAlias = jnp.ndarray
Centroid: TypeAlias = jnp.ndarray
Spread: TypeAlias = jnp.ndarray
Gradient: TypeAlias = jnp.ndarray

Skill: TypeAlias = jnp.ndarray

ExtraScores: TypeAlias = Dict[str, ArrayTree]

# Pareto fronts
T = TypeVar("T", bound=Union[Fitness, Genotype, Descriptor, jnp.ndarray])


class ParetoFront(Generic[T]):
    def __init__(self) -> None:
        super().__init__()


Mask: TypeAlias = jnp.ndarray

# Others
RNGKey: TypeAlias = jax.Array
Metrics: TypeAlias = Dict[str, jnp.ndarray]


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

    mean_observations: jnp.ndarray
    std_observations: jnp.ndarray

    @classmethod
    def create(
        cls,
        model_params: Params,
        mean_observations: jnp.ndarray,
        std_observations: jnp.ndarray,
    ) -> AuroraExtraInfoNormalization:
        return cls(
            model_params=model_params,
            mean_observations=mean_observations,
            std_observations=std_observations,
        )
