"""Defines some types used in QDax"""

from typing import Dict, Generic, TypeVar, Union

import brax.envs
import jax
import jax.numpy as jnp
import jumanji
from chex import ArrayTree
from typing_extensions import TypeAlias

JumanjiState: TypeAlias = ArrayTree
JumanjiTimeStep: TypeAlias = jumanji.types.TimeStep

# MDP types
Observation: TypeAlias = jnp.ndarray
Action: TypeAlias = jnp.ndarray
Reward: TypeAlias = jnp.ndarray
Done: TypeAlias = jnp.ndarray
EnvState: TypeAlias = brax.envs.State
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
