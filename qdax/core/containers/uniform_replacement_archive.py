from __future__ import annotations

import jax
import jax.numpy as jnp

from qdax.core.containers.archive import Archive
from qdax.types import RNGKey


class UniformReplacementArchive(Archive):
    """Stores jnp.ndarray and use a uniform replacement when the
    maximum size is reached.

    Instead of replacing elements in a FIFO manner, like the Archive,
    this implementation removes elements uniformly to replace them by
    the newly added ones.

    Most methods are inherited from Archive.
    """

    random_key: RNGKey

    @classmethod
    def create(  # type: ignore
        cls,
        acceptance_threshold: float,
        state_descriptor_size: int,
        max_size: int,
        random_key: RNGKey,
    ) -> Archive:
        """Create an Archive instance.

        This class method provides a convenient way to create the archive while
        keeping the __init__ function for more general way to init an archive.

        Args:
            acceptance_threshold: the minimal distance to a stored descriptor to
                be respected for a new descriptor to be added.
            state_descriptor_size: the number of elements in a state descriptor.
            max_size: the maximal size of the archive. In case of overflow, previous
                elements are replaced by new ones. Defaults to 80000.
            random_key: a key to handle random operations. Defaults to key with
                seed = 0.

        Returns:
            A newly initialized archive.
        """

        archive = super().create(
            acceptance_threshold,
            state_descriptor_size,
            max_size,
        )

        return archive.replace(random_key=random_key)  # type: ignore

    @jax.jit
    def _single_insertion(self, state_descriptor: jnp.ndarray) -> Archive:
        """Insert a single element.

        If the archive is not full yet, the new element replaces a fake
        border, if it is full, it replaces a random element from the archive.

        Args:
            state_descriptor: state descriptor to be added.

        Returns:
            Return the archive with the newly added element."""
        new_current_position = self.current_position + 1
        is_full = new_current_position >= self.max_size

        random_key, subkey = jax.random.split(self.random_key)
        random_index = jax.random.randint(
            subkey, shape=(1,), minval=0, maxval=self.max_size
        )

        index = jnp.where(condition=is_full, x=random_index, y=new_current_position)

        new_data = self.data.at[index].set(state_descriptor)

        return self.replace(  # type: ignore
            current_position=new_current_position, data=new_data, random_key=random_key
        )
