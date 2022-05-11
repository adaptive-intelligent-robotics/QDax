"""Defines some types used in PaRL"""

import jax
import jax.numpy as jnp
import tree
from chex import ArrayTree
from typing_extensions import TypeAlias

# MDP types
Observation: TypeAlias = jnp.ndarray
Action: TypeAlias = jnp.ndarray
Reward: TypeAlias = jnp.ndarray
Done: TypeAlias = jnp.ndarray
EnvState: TypeAlias = jnp.ndarray
Params: TypeAlias = ArrayTree

# Evolution types
StateDescriptor: TypeAlias = jnp.ndarray
Fitness: TypeAlias = jnp.ndarray
Genotype: TypeAlias = ArrayTree
Descriptor: TypeAlias = jnp.ndarray
Centroid: TypeAlias = jnp.ndarray

ExtraScores: TypeAlias = ArrayTree

# Others
RNGKey: TypeAlias = jax.random.KeyArray
Metrics: TypeAlias = tree.StructureKV[str, jnp.ndarray]
