"""Utils to handle pareto fronts."""

import jax
import jax.numpy as jnp


def compute_pareto_dominance(
    criteria_point: jnp.ndarray, batch_of_criteria: jnp.ndarray
) -> jnp.ndarray:
    """
    Returns if a point is pareto dominated given a set of points or not.
    We use maximization convention here.

    # criteria_point of shape (num_criteria,)
    # batch_of_criteria of shape (num_points, num_criteria)
    """
    diff = jnp.subtract(batch_of_criteria, criteria_point)
    return jnp.any(jnp.all(diff > 0, axis=-1))


def compute_pareto_front(batch_of_criteria: jnp.ndarray) -> jnp.ndarray:
    """
    Returns an array of boolean that states for each element if it is in
    the pareto front or not.

    # batch_of_criteria of shape (num_points, num_criteria)
    """
    func = jax.vmap(lambda x: ~compute_pareto_dominance(x, batch_of_criteria))
    return func(batch_of_criteria)


def compute_masked_pareto_dominance(
    criteria_point: jnp.ndarray, batch_of_criteria: jnp.ndarray, mask: jnp.ndarray
) -> jnp.ndarray:
    """
    Returns if a point is pareto dominated given a set of points or not.
    We use maximization convention here.
    This function is to be used with constant size batches of criteria,
    thus a mask is used to know which values are padded.

    # criteria_point of shape (num_criteria,)
    # batch_of_criteria of shape (batch_size, num_criteria)
    # mask of shape (batch_size,), 1.0 where there is not element, 0 otherwise
    """

    diff = jnp.subtract(batch_of_criteria, criteria_point)
    neutral_values = -jnp.ones_like(diff)
    diff = jax.vmap(lambda x1, x2: jnp.where(mask, x1, x2), in_axes=(1, 1), out_axes=1)(
        neutral_values, diff
    )
    return jnp.any(jnp.all(diff > 0, axis=-1))


def compute_masked_pareto_front(
    batch_of_criteria: jnp.ndarray, mask: jnp.ndarray
) -> jnp.ndarray:
    """
    Returns an array of boolean that states for each element if it is to be
    considered or not. This function works is to be used constant size batches of
    criteria, thus a mask is used to know which values are padded.
    """
    func = jax.vmap(
        lambda x: ~compute_masked_pareto_dominance(x, batch_of_criteria, mask)
    )
    return func(batch_of_criteria) * ~mask


def compute_hypervolume(
    pareto_front: jnp.ndarray, reference_point: jnp.ndarray
) -> jnp.ndarray:
    """Compute hypervolume of a pareto front.

    TODO: implement recursive version of
    https://github.com/anyoptimization/pymoo/blob/master/pymoo/vendor/hv.py
    """

    num_objectives = pareto_front.shape[1]

    assert (
        num_objectives == 2
    ), "Hypervolume calculation for more than 2 objectives not yet supported."

    pareto_front = jnp.concatenate(  # type: ignore
        (pareto_front, jnp.expand_dims(reference_point, axis=0)), axis=0
    )
    idx = jnp.argsort(pareto_front[:, 0])
    mask = pareto_front[idx, :] != -jnp.inf
    sorted_front = (pareto_front[idx, :] - reference_point) * mask
    sumdiff = (sorted_front[1:, 0] - sorted_front[:-1, 0]) * sorted_front[1:, 1]
    sumdiff = sumdiff * mask[:-1, 0]
    hypervolume = jnp.sum(sumdiff)

    return hypervolume
