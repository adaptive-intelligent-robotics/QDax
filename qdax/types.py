"""Defines some types used in PaRL"""
from typing import Any, Mapping, Union

import jax.numpy as jnp
import tree
from typing_extensions import TypeAlias

# MDP types
Observation: TypeAlias = jnp.ndarray
Action: TypeAlias = jnp.ndarray
Reward: TypeAlias = jnp.ndarray
Done: TypeAlias = jnp.ndarray
EnvState: TypeAlias = jnp.ndarray
Params: TypeAlias = Union[jnp.ndarray, Mapping[str, Mapping[str, Any]]]

# Evolution types
StateDescriptor: TypeAlias = jnp.ndarray
Fitness: TypeAlias = jnp.ndarray
Genotype: TypeAlias = jnp.ndarray
Descriptor: TypeAlias = jnp.ndarray
Centroid: TypeAlias = jnp.ndarray
EmitterState: TypeAlias = tree.StructureKV[str, jnp.ndarray]

# Others
RNGKey: TypeAlias = Union[Any, jnp.ndarray]
Metrics: TypeAlias = tree.StructureKV[str, jnp.ndarray]
