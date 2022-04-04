"""Defines some types used in PaRL"""
from typing import Any, NewType

import jax.numpy as jnp
import tree

# MDP types
Observation = NewType("Observation", jnp.ndarray)
Action = NewType("Action", jnp.ndarray)
Reward = NewType("Reward", jnp.ndarray)
Done = NewType("Done", jnp.ndarray)
EnvState = NewType("Env_State", Any)

# Controller types
Params = NewType("Params", Any)
StateDescriptor = NewType("StateDescriptor", jnp.ndarray)

Scores = NewType("Scores", jnp.ndarray)
Fitness = NewType("Fitness", jnp.ndarray)
Genotypes = NewType("Genotypes", jnp.ndarray)
Descriptors = NewType("Descriptors", jnp.ndarray)

# MAP Elites types
Centroids = NewType("Centroids", jnp.ndarray)
EmitterState = tree.StructureKV[str, jnp.ndarray]

# Others
RNGKey = NewType("RNGKey", jnp.ndarray)
Metrics = tree.StructureKV[str, jnp.ndarray]
