import jax
import jax.numpy as jnp

from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.types import Descriptor, Params
from qdax.utils import train_seq2seq


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


def get_aurora_bd(
    data: QDTransition,
    mask: jnp.ndarray,
    model_params: Params,
    mean_observations: jnp.ndarray,
    std_observations: jnp.ndarray,
    option: str = "full",
    hidden_size: int = 10,
    padding: bool = False,
) -> Descriptor:
    """Compute final aurora embedding.

    This function suppose that state descriptor is the xy position, as it
    just select the final one of the state descriptors given.
    """
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    state_obs = data.obs[:, ::10, :25]
    filtered_mask = mask[:, ::10, :]

    # add the x/y position - (batch_size, traj_length, 2)
    state_desc = data.state_desc[:, ::10]

    print("State Observations: ", state_obs)
    print("XY positions: ", state_desc)

    if option == "full":
        observations = jnp.concatenate([state_desc, state_obs], axis=-1)
        print("New observations: ", observations)
    elif option == "no_sd":
        observations = state_obs
    elif option == "only_sd":
        observations = state_desc

    # add padding when the episode is done
    if padding:
        observations = jnp.where(filtered_mask, x=jnp.array(0.0), y=observations)

    # lstm seq2seq
    model = train_seq2seq.get_model(observations.shape[-1], True, hidden_size)
    normalized_observations = (observations - mean_observations) / std_observations
    descriptors = model.apply(
        {"params": model_params}, normalized_observations, method=model.encode
    )

    print("Observations out of get aurora bd: ", observations)

    return descriptors.squeeze(), observations.squeeze()
