"""Defines an unstructured archive and a euclidean novelty scorer."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode


class Archive(PyTreeNode):
    """Stores jnp.ndarray in a way that makes insertion jittable.

    An example of use of the archive is the algorithm QDPG: state
    descriptors are stored in this archive and a novelty scorer compares
    new state desciptors to the state descriptors stored in this archive.

    Note: notations suppose that the elements are called state desciptors.
    If we where to use this structure for another application, it would be
    better to change the variables name for another one. Does not seem
    necessary at the moment though.
    """

    data: jnp.ndarray  # initialised with nan everywhere
    current_position: int
    acceptance_threshold: float
    state_descriptor_size: int
    max_size: int

    @property
    def size(self) -> float:
        """Compute the number of state descriptors stored in the archive.

        Returns:
            Size of the archive.
        """
        # remove fake borders
        fake_data = jnp.isnan(self.data)

        # count number of real data
        return sum(~fake_data)

    @classmethod
    def create(
        cls,
        acceptance_threshold: float,
        state_descriptor_size: int,
        max_size: int,
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

        Returns:
            A newly initialized archive.
        """
        init_data = jnp.ones((max_size, state_descriptor_size)) * jnp.nan
        return cls(  # type: ignore
            data=init_data,
            current_position=0,
            acceptance_threshold=acceptance_threshold,
            state_descriptor_size=state_descriptor_size,
            max_size=max_size,
        )

    @jax.jit
    def _single_insertion(self, state_descriptor: jnp.ndarray) -> Archive:
        """Insert a single element.

        If the archive is not full yet, the new element replaces a fake
        border, if it is full, it replaces the element that was inserted
        first in the archive.

        Args:
            state_descriptor: state descriptor to be added.

        Returns:
            Return the archive with the newly added element.
        """
        new_current_position = self.current_position + 1
        new_data = jax.lax.dynamic_update_slice_in_dim(
            self.data,
            state_descriptor.reshape(1, -1),
            start_index=self.current_position % self.max_size,
            axis=0,
        )

        return self.replace(  # type: ignore
            current_position=new_current_position, data=new_data
        )

    @jax.jit
    def _conditioned_single_insertion(
        self, condition: bool, state_descriptor: jnp.ndarray
    ) -> Tuple[Archive, jnp.ndarray]:
        """Inserts a single element under a condition.

        The function also retrieves the added elements.

        Args:
            condition: condition for being added in the archive.
            state_descriptor: state descriptor to be added under the
                given condition.

        Returns:
            The new archive and the added elements.
        """

        def true_fun(
            archive: Archive, state_descriptor: jnp.ndarray
        ) -> Tuple[Archive, jnp.ndarray]:
            return archive._single_insertion(state_descriptor), state_descriptor

        def false_fun(
            archive: Archive, state_descriptor: jnp.ndarray
        ) -> Tuple[Archive, jnp.ndarray]:
            return archive, jnp.ones_like(state_descriptor) * jnp.nan

        return jax.lax.cond(  # type: ignore
            condition, true_fun, false_fun, self, state_descriptor
        )

    @jax.jit
    def insert(self, state_descriptors: jnp.ndarray) -> Archive:
        """Tries to insert a batch of state descriptors in the archive.

        1. First, look at the distance of each new state descriptor with the
        already stored ones.
        2. Then, scan the state descriptors, check the distance with
        the descriptors inserted during the scan.
        3. If the state descriptor verified the first condition (not too close
        to a state descriptor in the old archive) and the second (not too close
        from a state descriptor that has just been added), then it is added
        to the archive.

        Note 1: the archive has a fixed size, hence, in case of overflow, the
        first elements added are removed first (FIFO style).
        Note 2: keep in mind that fake descriptors are used to help keep the size
        constant. Those correspond to a descriptor very far away from the typical
        values of the problem at hand.

        Args:
            state_descriptors: state descriptors to be added.

        Returns:
            New archive updated with the state descriptors.
        """
        state_descriptors = state_descriptors.reshape((-1, state_descriptors.shape[-1]))

        # get nearest neigbor for each new state descriptor
        values, _indices = knn(self.data, state_descriptors, 1)

        # get indices where distance bigger than threshold
        relevant_indices = jnp.where(
            values.squeeze() > self.acceptance_threshold, x=0, y=1
        )

        def iterate_fn(
            carry: Tuple[Archive, jnp.ndarray, int], condition_data: Dict
        ) -> Tuple[Tuple[Archive, jnp.ndarray, int], Any]:
            """Iterates over the archive to add elements one after the other.

            Args:
                carry: tuple containing the archive, the state descriptors and the
                    indices.

                condition_data: the first addition condition of the state descriptors
                    given, which corresponds to being sufficiently far away from already
                    stored descriptors.

            Returns:
                The update tuple.
            """
            archive, new_elements, index = carry

            first_condition = condition_data["condition"]
            state_descriptor = condition_data["state_descriptor"]

            # do the filtering among the added elements
            # get nearest neigbor for each new state descriptor
            values, _indices = knn(new_elements, state_descriptor.reshape(1, -1), 1)

            # get indices where distance bigger than threshold
            not_too_close = jnp.where(
                values.squeeze() > self.acceptance_threshold, x=0, y=1
            )
            second_condition = not_too_close.sum()
            condition = (first_condition + second_condition) == 0

            new_archive, added_element = archive._conditioned_single_insertion(
                condition, state_descriptor
            )
            new_elements = new_elements.at[index].set(added_element)
            index += 1

            return (
                (new_archive, new_elements, index),
                (),
            )

        new_elements = jnp.ones_like(state_descriptors) * jnp.nan

        # iterate over the indices
        (new_archive, _, _), _ = jax.lax.scan(
            iterate_fn,
            (self, new_elements, 0),
            {
                "condition": relevant_indices,
                "state_descriptor": state_descriptors,
            },
        )

        return new_archive  # type: ignore


def score_euclidean_novelty(
    archive: Archive,
    state_descriptors: jnp.ndarray,
    num_nearest_neighb: int,
    scaling_ratio: float,
) -> jnp.ndarray:
    """Scores the novelty of a jnp.ndarray with respect to the elements of an archive.

    Typical use case in the construction of the diversity rewards
    in QDPG.

    Args:
        archive: an archive of state descriptors.
        state_descriptors: state descriptors which novelty must be scored.
        num_nearest_neighb: the number of nearest neighbors to be considered
            when scoring.
        scaling_ratio: the ratio applied to the the mean distance to obtain the
            final value.

    Returns:
        The novelty scores of the given state descriptors.
    """
    values, _indices = knn(archive.data, state_descriptors, num_nearest_neighb)

    summed_distances = jnp.mean(jnp.square(values), axis=1)
    return scaling_ratio * summed_distances


@partial(jax.jit, static_argnames=("k"))
def knn(
    data: jnp.ndarray, new_data: jnp.ndarray, k: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """K nearest neigbors - Brute force implementation.
    Using euclidean distance.

    Code from https://www.kernel-operations.io/keops/_auto_benchmarks/
    plot_benchmark_KNN.html

    Args:
        data: given reference data.
        new_data: data to be compared to the reference data.
        k: number of neigbors to consider.

    Returns:
        The distances and indices of the nearest neighbors.
    """

    # compute distances
    dist = (
        (new_data**2).sum(-1)[:, None]
        + (data**2).sum(-1)[None, :]
        - 2 * new_data @ data.T
    )

    dist = jnp.nan_to_num(dist, nan=jnp.inf)

    # clipping necessary - numerical approx make some distancies negative
    dist = jnp.sqrt(jnp.clip(dist, a_min=0.0))

    # return values, indices
    values, indices = qdax_top_k(-dist, k)

    return -values, indices


@partial(jax.jit, static_argnames=("k"))
def qdax_top_k(data: jnp.ndarray, k: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the top k elements of an array.

    Interestingly, this naive implementation is faster than the native implementation
    of jax for small k. See issue: https://github.com/google/jax/issues/9940

    Waiting for updates in jax to change this implementation.

    Args:
        data: given data.
        k: number of top elements to determine.

    Returns:
        The values of the elements and their indices in the array.
    """

    def top_1(data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        indice = jnp.argmax(data, axis=1)
        value = jax.vmap(lambda x, y: x[y])(data, indice)
        data = jax.vmap(lambda x, y: x.at[y].set(-jnp.inf))(data, indice)

        return data, value, indice

    def scannable_top_1(
        carry: jnp.ndarray, unused: Any
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        data = carry
        data, value, indice = top_1(data)
        return data, (value, indice)

    data, (values, indices) = jax.lax.scan(scannable_top_1, data, (), k)

    return values.T, indices.T
