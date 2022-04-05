import jax
import jax.numpy as jnp

from qdax.types import Descriptor
from qdax.utils.mdp_utils import Transition


def get_final_xy_position(data: Transition, mask: jnp.ndarray) -> Descriptor:
    """
    This function is sued to map state descriptors to behavior descriptors.
    Compute final xy position. This function assumes that state descriptor is the xy
    position, as it just selects the final one of the state descriptors given.

    Args:
        data: a trajectory of transitions
        mask: a mask over the fixed size trajectory to know when the episode ended. The
            mask contains true when the episode is not done and false otherwise.

    Returns:
        the final xy position of the agent
    """
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    # Get behavior descriptor
    last_index = jnp.int32(jnp.sum(1.0 - mask, axis=1)) - 1
    descriptors = jax.vmap(lambda x, y: x[y])(data.state_desc, last_index)

    return descriptors.squeeze()


def get_feet_contact_proportion(data: Transition, mask: jnp.ndarray) -> Descriptor:
    """
    This function is sued to map state descriptors to behavior descriptors.
    Compute feet contact time proportion. This function assumes that state descriptor
    is the feet contact, as it just computes the mean of the state descriptors given.

    Args:
        data: a trajectory of transitions
        mask: a mask over the fixed size trajectory to know when the episode ended. The
            mask contains true when the episode is not done and false otherwise.

    Returns:
        the feet contacts frequency over the trajectory.
    """
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    # Get behavior descriptor
    descriptors = jnp.sum(data.state_desc * (1.0 - mask), axis=1)
    descriptors = descriptors / jnp.sum(1.0 - mask, axis=1)

    return descriptors
