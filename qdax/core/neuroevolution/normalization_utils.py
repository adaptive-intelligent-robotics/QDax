"""Utilities functions to perform normalization (generally on observations in RL)."""


from typing import NamedTuple

import jax.numpy as jnp

from qdax.types import Observation


class RunningMeanStdState(NamedTuple):
    """Running statistics for observtations/rewards"""

    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def update_running_mean_std(
    std_state: RunningMeanStdState, obs: Observation
) -> RunningMeanStdState:
    """Update running statistics with batch of observations (Welford's algorithm)"""

    running_mean, running_variance, normalization_steps = std_state

    step_increment = obs.shape[0]

    total_new_steps = normalization_steps + step_increment

    # Compute the incremental update and divide by the number of new steps.
    input_to_old_mean = obs - running_mean
    mean_diff = jnp.sum(input_to_old_mean / total_new_steps, axis=0)
    new_mean = running_mean + mean_diff

    # Compute difference of input to the new mean for Welford update.
    input_to_new_mean = obs - new_mean
    var_diff = jnp.sum(input_to_new_mean * input_to_old_mean, axis=0)

    return RunningMeanStdState(new_mean, running_variance + var_diff, total_new_steps)


def normalize_with_rmstd(
    obs: jnp.ndarray,
    rmstd: RunningMeanStdState,
    std_min_value: float = 1e-6,
    std_max_value: float = 1e6,
    apply_clipping: bool = True,
) -> jnp.ndarray:
    """Normalize input with provided running statistics"""

    running_mean, running_variance, normalization_steps = rmstd
    variance = running_variance / (normalization_steps + 1.0)
    # We clip because the running_variance can become negative,
    # presumably because of numerical instabilities.
    if apply_clipping:
        variance = jnp.clip(variance, std_min_value, std_max_value)
        return jnp.clip((obs - running_mean) / jnp.sqrt(variance), -5, 5)
    else:
        return (obs - running_mean) / jnp.sqrt(variance)
