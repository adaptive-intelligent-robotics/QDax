"""Defines some types used in PaRL"""
from typing import Any, NewType

import jax.numpy as jnp
import tree

# MDP types
Observation = NewType("Observation", jnp.ndarray)
Action = NewType("Action", jnp.ndarray)
Reward = NewType("Reward", jnp.ndarray)
Done = NewType("Done", jnp.ndarray)
EnvState = NewType("EnvState", Any)
Params = NewType("Params", Any)

# Evolution types
StateDescriptor = NewType("StateDescriptor", jnp.ndarray)
Fitness = NewType("Fitness", jnp.ndarray)
Genotype = NewType("Genotypes", jnp.ndarray)
Descriptor = NewType("Descriptors", jnp.ndarray)
Centroid = NewType("Centroids", jnp.ndarray)
EmitterState = tree.StructureKV[str, jnp.ndarray]

# Others
RNGKey = NewType("RNGKey", jnp.ndarray)
Metrics = tree.StructureKV[str, jnp.ndarray]
