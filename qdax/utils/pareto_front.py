"""Utils to handle pareto fronts."""

import chex
import jax
import jax.numpy as jnp

from qdax.types import Mask, ParetoFront


def compute_pareto_dominance(
    criteria_point: jnp.ndarray, batch_of_criteria: jnp.ndarray
) -> jnp.ndarray:
    """Returns if a point is pareto dominated given a set of points or not.
    We use maximization convention here.

    criteria_point has shape (num_criteria,)
    batch_of_criteria has shape (num_points, num_criteria)

    Args:
        criteria_point: a vector of values.
        batch_of_criteria: a batch of vector of values.

    Returns:
        Return booleans when the vector is dominated by the batch.
    """
    diff = jnp.subtract(batch_of_criteria, criteria_point)
    return jnp.any(jnp.all(diff > 0, axis=-1))


def compute_pareto_front(batch_of_criteria: jnp.ndarray) -> jnp.ndarray:
    """Returns an array of boolean that states for each element if it is
    in the pareto front or not.

    Args:
        batch_of_criteria: a batch of points of shape (num_points, num_criteria)

    Returns:
        An array of boolean with the boolean stating if each point is on the
        front or not.
    """
    func = jax.vmap(lambda x: ~compute_pareto_dominance(x, batch_of_criteria))
    return func(batch_of_criteria)


def compute_masked_pareto_dominance(
    criteria_point: jnp.ndarray, batch_of_criteria: jnp.ndarray, mask: Mask
) -> jnp.ndarray:
    """Returns if a point is pareto dominated given a set of points or not.
    We use maximization convention here.

    This function is to be used with constant size batches of criteria,
    thus a mask is used to know which values are padded.

    Args:
        criteria_point: values to be evaluated, of shape (num_criteria,)
        batch_of_criteria: set of points to compare with,
            of shape (batch_size, num_criteria)
        mask: mask of shape (batch_size,), 1.0 where there is not element,
            0 otherwise

    Returns:
        Boolean assessing if the point is dominated or not.
    """

    diff = jnp.subtract(batch_of_criteria, criteria_point)
    neutral_values = -jnp.ones_like(diff)
    diff = jax.vmap(lambda x1, x2: jnp.where(mask, x1, x2), in_axes=(1, 1), out_axes=1)(
        neutral_values, diff
    )
    return jnp.any(jnp.all(diff > 0, axis=-1))


def compute_masked_pareto_front(
    batch_of_criteria: jnp.ndarray, mask: Mask
) -> jnp.ndarray:
    """Returns an array of boolean that states for each element if it is to be
    considered or not. This function is to be used with batches of constant size
    criteria, thus a mask is used to know which values are padded.

    Args:
        batch_of_criteria: data points considered
        mask: mask to hide several points

    Returns:
        An array of boolean stating the points to consider.
    """
    func = jax.vmap(
        lambda x: ~compute_masked_pareto_dominance(x, batch_of_criteria, mask)
    )
    return func(batch_of_criteria) * ~mask


def compute_hypervolume(
    pareto_front: ParetoFront[jnp.ndarray], reference_point: jnp.ndarray
) -> jnp.ndarray:
    """Compute hypervolume of a pareto front.

    Args:
        pareto_front: a pareto front, shape (pareto_size, num_objectives)
        reference_point: a reference point to compute the volume, of shape
            (num_objectives,)

    Returns:
        The hypervolume of the pareto front.
    """
    # check the number of objectives
    custom_message = (
        "Hypervolume calculation for more than" " 2 objectives not yet supported."
    )
    chex.assert_axis_dimension(
        tensor=pareto_front,
        axis=1,
        expected=2,
        custom_message=custom_message,
    )

    # concatenate the reference point to prepare for the area computation
    pareto_front = jnp.concatenate(  # type: ignore
        (pareto_front, jnp.expand_dims(reference_point, axis=0)), axis=0
    )
    # get ordered indices for the first objective
    idx = jnp.argsort(pareto_front[:, 0])
    # Note: this orders it in inversely for the second objective

    # create the mask - hide fake elements (those having -inf fitness)
    mask = pareto_front[idx, :] != -jnp.inf

    # sort the front and offset it with the ref point
    sorted_front = (pareto_front[idx, :] - reference_point) * mask

    # compute area rectangles between successive points
    sumdiff = (sorted_front[1:, 0] - sorted_front[:-1, 0]) * sorted_front[1:, 1]

    # remove the irrelevant values - where a mask was applied
    sumdiff = sumdiff * mask[:-1, 0]

    # get the hypervolume by summing the succcessives areas
    hypervolume = jnp.sum(sumdiff)

    return hypervolume
