from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


def arm(params: Genotype) -> Tuple[Fitness, Descriptor]:
    """
    Compute the fitness and descriptor of one individual in the Planar Arm task.
    Based on the Planar Arm implementation in fast_map_elites
    (https://github.com/hucebot/fast_map-elites).

    Args:
        params: genotype of the individual to evaluate, corresponding to
            the normalised angles for each DoF of the arm.
            Params should be between [0, 1].

    Returns:
        f: the fitness of the individual, given as the variance of the angles.
        descriptor: the descriptor of the individual, given as the [x, y] position
            of the end-effector of the arm.
            Descriptor is normalized to [0, 1] regardless of the DoF.
            Arm is centered at 0.5, 0.5.
    """

    x = jnp.clip(params, 0, 1)
    size = params.shape[0]

    f = jnp.sqrt(jnp.mean(jnp.square(x - jnp.mean(x))))

    # Compute the end-effector position - forward kinemateics
    cum_angles = jnp.cumsum(2 * jnp.pi * x - jnp.pi)
    x_pos = jnp.sum(jnp.cos(cum_angles)) / (2 * size) + 0.5
    y_pos = jnp.sum(jnp.sin(cum_angles)) / (2 * size) + 0.5

    return -f, jnp.array([x_pos, y_pos])


def arm_scoring_function(
    params: Genotype,
    key: RNGKey,
) -> Tuple[Fitness, Descriptor, ExtraScores]:
    """
    Evaluate policies contained in params in parallel.
    """
    fitnesses, descriptors = jax.vmap(arm)(params)

    return fitnesses, descriptors, {}


def noisy_arm_scoring_function(
    params: Genotype,
    key: RNGKey,
    fit_variance: float,
    desc_variance: float,
    params_variance: float,
) -> Tuple[Fitness, Descriptor, ExtraScores]:
    """
    Evaluate policies contained in params in parallel.
    """

    key, f_subkey, d_subkey, p_subkey = jax.random.split(key, num=4)

    # Add noise to the parameters
    params = params + jax.random.normal(p_subkey, shape=params.shape) * params_variance

    # Evaluate
    fitnesses, descriptors = jax.vmap(arm)(params)

    # Add noise to the fitnesses and descriptors
    fitnesses = (
        fitnesses + jax.random.normal(f_subkey, shape=fitnesses.shape) * fit_variance
    )
    descriptors = (
        descriptors
        + jax.random.normal(d_subkey, shape=descriptors.shape) * desc_variance
    )

    return fitnesses, descriptors, {}


def parameterized_arm_end_effector(
    params: Genotype, task_descriptor: Descriptor
) -> jax.Array:
    """
    Compute the end-effector position of the parameterized Planar Arm
    introduced in Mouret and Maguire (GECCO 2020,
    https://arxiv.org/abs/2003.04407).

    The morphology of the arm is described by the task descriptor
    [link_length, max_angle], both normalized to [0, 1]. The link lengths
    and the joint limits are divided by the number of DoF so that the total
    length of the arm (link_length, in [0, 1]) and its reaching abilities do
    not depend on the dimensionality.

    Args:
        params: genotype of the individual to evaluate, corresponding to
            the normalised angles for each DoF of the arm.
            Params should be between [0, 1].
        task_descriptor: [link_length, max_angle] descriptor of the
            morphology of the arm, in [0, 1]^2.

    Returns:
        the [x, y] position of the end-effector of the arm, in [-1, 1]^2.
        Arm is centered at 0, 0.
    """
    link_length, max_angle = task_descriptor[0], task_descriptor[1]

    x = jnp.clip(params, 0, 1)
    size = params.shape[0]

    # Compute the end-effector position - forward kinematics
    angles = (x - 0.5) * max_angle * 2 * jnp.pi / size
    cum_angles = jnp.cumsum(angles)
    x_pos = jnp.sum(jnp.cos(cum_angles)) * link_length / size
    y_pos = jnp.sum(jnp.sin(cum_angles)) * link_length / size

    return jnp.array([x_pos, y_pos])


def multi_task_arm(
    params: Genotype,
    task_descriptor: Descriptor,
    target: jax.Array,
) -> Fitness:
    """
    Compute the fitness of one individual on one task of the multi-task
    Planar Arm family (Mouret and Maguire, GECCO 2020): the fitness is the
    negative distance between the end-effector of the arm, whose morphology
    is described by the task descriptor, and a fixed target.

    Args:
        params: genotype of the individual to evaluate, corresponding to
            the normalised angles for each DoF of the arm.
            Params should be between [0, 1].
        task_descriptor: [link_length, max_angle] descriptor of the
            morphology of the arm, in [0, 1]^2.
        target: [x, y] position of the target to reach.

    Returns:
        f: the fitness of the individual on the task, given as the negative
            Euclidean distance between the end-effector and the target.
    """
    end_effector = parameterized_arm_end_effector(params, task_descriptor)
    return -jnp.linalg.norm(end_effector - target)


def multi_task_arm_scoring_function(
    params: Genotype,
    task_descriptors: Descriptor,
    key: RNGKey,
    target: Tuple[float, float] = (1.0, 1.0),
) -> Tuple[Fitness, ExtraScores]:
    """
    Evaluate policies contained in params in parallel, each on its own task.
    To be used with Multi-task MAP-Elites.

    Args:
        params: genotypes of the individuals to evaluate, of shape
            (batch_size, num_dofs).
        task_descriptors: descriptors of the tasks on which the individuals
            are evaluated, of shape (batch_size, 2).
        key: unused jax PRNG random key, kept for API consistency.
        target: [x, y] position of the target to reach. The default target
            (1, 1) is the one used in the paper.

    Returns:
        the fitness of each individual on its task and an empty dict of
        extra scores.
    """
    scoring_fn = partial(multi_task_arm, target=jnp.asarray(target))
    fitnesses = jax.vmap(scoring_fn)(params, task_descriptors)

    return fitnesses, {}


def multi_task_multi_behavior_arm(
    params: Genotype, task_descriptor: Descriptor
) -> Tuple[Fitness, Descriptor]:
    """
    Compute the fitness and behavior descriptor of one individual on one
    task of the multi-task Planar Arm family. As in the standard Planar Arm
    task, the fitness is the negative variance of the angles and the
    behavior descriptor is the position of the end-effector of the arm,
    whose morphology is described by the task descriptor.

    Args:
        params: genotype of the individual to evaluate, corresponding to
            the normalised angles for each DoF of the arm.
            Params should be between [0, 1].
        task_descriptor: [link_length, max_angle] descriptor of the
            morphology of the arm, in [0, 1]^2.

    Returns:
        f: the fitness of the individual, given as the variance of the
            angles.
        descriptor: the descriptor of the individual, given as the [x, y]
            position of the end-effector of the arm.
            Descriptor is normalized to [0, 1] regardless of the morphology.
            Arm is centered at 0.5, 0.5.
    """
    x = jnp.clip(params, 0, 1)
    f = jnp.sqrt(jnp.mean(jnp.square(x - jnp.mean(x))))

    end_effector = parameterized_arm_end_effector(params, task_descriptor)
    descriptor = end_effector / 2.0 + 0.5

    return -f, descriptor


def multi_task_multi_behavior_arm_scoring_function(
    params: Genotype,
    task_descriptors: Descriptor,
    key: RNGKey,
) -> Tuple[Fitness, Descriptor, ExtraScores]:
    """
    Evaluate policies contained in params in parallel, each on its own task.
    To be used with Multi-Task Multi-Behavior MAP-Elites.

    Args:
        params: genotypes of the individuals to evaluate, of shape
            (batch_size, num_dofs).
        task_descriptors: descriptors of the tasks on which the individuals
            are evaluated, of shape (batch_size, 2).
        key: unused jax PRNG random key, kept for API consistency.

    Returns:
        the fitness and behavior descriptor of each individual on its task
        and an empty dict of extra scores.
    """
    fitnesses, descriptors = jax.vmap(multi_task_multi_behavior_arm)(
        params, task_descriptors
    )

    return fitnesses, descriptors, {}
