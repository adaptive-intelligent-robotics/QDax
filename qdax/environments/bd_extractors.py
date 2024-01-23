from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp

from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.types import Descriptor, Params


def get_final_xy_position(data: QDTransition, mask: jnp.ndarray) -> Descriptor:
    """Compute final xy positon.

    This function suppose that state descriptor is the xy position, as it
    just select the final one of the state descriptors given.
    """
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    # Get behavior descriptor
    last_index = jnp.int32(jnp.sum(1.0 - mask, axis=1)) - 1
    descriptors = jax.vmap(lambda x, y: x[y])(data.state_desc, last_index)

    # remove the dim coming from the trajectory
    return descriptors.squeeze(axis=1)


def get_feet_contact_proportion(data: QDTransition, mask: jnp.ndarray) -> Descriptor:
    """Compute feet contact time proportion.

    This function suppose that state descriptor is the feet contact, as it
    just computes the mean of the state descriptors given.
    """
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    # Get behavior descriptor
    descriptors = jnp.sum(data.state_desc * (1.0 - mask), axis=1)
    descriptors = descriptors / jnp.sum(1.0 - mask, axis=1)

    return descriptors


class AuroraExtraInfo(flax.struct.PyTreeNode):
    """
    Information specific to the AURORA algorithm.

    Args:
        model_params: the parameters of the dimensionality reduction model
    """

    model_params: Params


class AuroraExtraInfoNormalization(AuroraExtraInfo):
    """
    Information specific to the AURORA algorithm. In particular, it contains
    the normalization parameters for the observations.

    Args:
        model_params: the parameters of the dimensionality reduction model
        mean_observations: the mean of observations
        std_observations: the std of observations
    """

    mean_observations: jnp.ndarray
    std_observations: jnp.ndarray

    @classmethod
    def create(
        cls,
        model_params: Params,
        mean_observations: jnp.ndarray,
        std_observations: jnp.ndarray,
    ) -> AuroraExtraInfoNormalization:
        return cls(
            model_params=model_params,
            mean_observations=mean_observations,
            std_observations=std_observations,
        )


def get_aurora_encoding(
    observations: jnp.ndarray,
    aurora_extra_info: AuroraExtraInfoNormalization,
    model: flax.linen.Module,
) -> Descriptor:
    """
    Compute final aurora embedding.

    This function suppose that state descriptor is the xy position, as it
    just select the final one of the state descriptors given.
    """
    model_params = aurora_extra_info.model_params
    mean_observations = aurora_extra_info.mean_observations
    std_observations = aurora_extra_info.std_observations

    # lstm seq2seq
    normalized_observations = (observations - mean_observations) / std_observations
    descriptors = model.apply(
        {"params": model_params}, normalized_observations, method=model.encode
    )

    return descriptors.squeeze()
