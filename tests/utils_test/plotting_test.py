from typing import Tuple

import jax
import jax.numpy as jnp
import pytest

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.utils.plotting import plot_multidimensional_map_elites_grid


@pytest.mark.parametrize(
    "num_descriptors, grid_shape",
    [
        (1, (4,)),
        (
            2,
            (
                4,
                4,
            ),
        ),
        (
            3,
            (
                4,
                4,
                4,
            ),
        ),
        (3, (4, 4, 2)),
        (4, (4, 4, 4, 4)),
    ],
)
def test_onion_grid(num_descriptors: int, grid_shape: Tuple[int, ...]) -> None:
    num_descriptors = num_descriptors
    grid_shape = grid_shape

    minval = jnp.array([0] * num_descriptors)
    maxval = jnp.array([1] * num_descriptors)

    random_key = jax.random.PRNGKey(seed=0)
    random_key, key_desc, key_fit = jax.random.split(random_key, num=3)

    number_samples_test = 300
    descriptors = jax.random.uniform(
        key_desc, shape=(number_samples_test, num_descriptors)
    )
    fitnesses = jax.random.uniform(key_fit, shape=(number_samples_test,))
    # Uncomment to test with "empty" descriptors
    # fitnesses = fitnesses.at[10:].set(-jnp.inf)

    repertoire = MapElitesRepertoire(None, fitnesses, descriptors, None)

    fig, ax = plot_multidimensional_map_elites_grid(
        repertoire=repertoire,
        minval=minval,
        maxval=maxval,
        grid_shape=grid_shape,
        vmin=0,
        vmax=1,
    )

    pytest.assume(fig is not None)


if __name__ == "__main__":
    test_onion_grid(num_descriptors=4, grid_shape=(4, 4, 4, 4))
