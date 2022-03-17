from typing import Any, NewType

import jax
import jax.numpy as jnp

# Define Solutions
Scores = NewType("Scores", jnp.ndarray)
Fitness = NewType("Fitness", jnp.ndarray)
Genotypes = NewType("Genotypes", jnp.ndarray)
Descriptors = NewType("Descriptors", jnp.ndarray)

jax.tree_util.register_pytree_node(Genotypes, lambda s: ((s), None), lambda _, xs: xs)
jax.tree_util.register_pytree_node(Scores, lambda s: ((s), None), lambda _, xs: xs)
jax.tree_util.register_pytree_node(Descriptors, lambda s: ((s), None), lambda _, xs: xs)
jax.tree_util.register_pytree_node(Fitness, lambda s: ((s), None), lambda _, xs: xs)

RNGKey = NewType("RNGKey", jnp.ndarray)

Centroids = NewType("Centroids", jnp.ndarray)
Grid = NewType("Grid", jnp.ndarray)
GridScores = NewType("Grid_Scores", jnp.ndarray)
GridDescriptors = NewType("Grid_Descriptors", jnp.ndarray)


Observation = NewType("Observation", jnp.ndarray)
Action = NewType("Action", jnp.ndarray)
Reward = NewType("Reward", jnp.ndarray)
Done = NewType("Done", jnp.ndarray)
StateDescriptor = NewType("StateDescriptor", jnp.ndarray)
Skill = NewType("Skill", jnp.ndarray)
TrainingState = NewType("TrainingState", Any)

Transition = NewType("Transition", Any)

Params = NewType("Params", Any)
