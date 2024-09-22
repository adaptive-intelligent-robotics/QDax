from typing import Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.core.emitters.selectors.abstract_selector import Selector
from qdax.custom_types import Genotype, RNGKey


class NoveltySelector(Selector):
    def __init__(self, num_nn: int):
        self._num_nn = num_nn

    def select(
        self,
        number_parents_to_select: int,
        repertoire: Repertoire,
        emitter_state: EmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, EmitterState, RNGKey]:
        """
        Novelty-based selection of parents
        """

        repertoire_genotypes = repertoire.genotypes
        fitnesses = repertoire.fitnesses
        descriptors = repertoire.descriptors

        num_genotypes = descriptors.shape[0]
        repertoire_empty = fitnesses == -jnp.inf

        # calculate novelty score using the k-nearest-neighbors
        v_dist = jax.vmap(lambda x, y: jnp.linalg.norm(x - y), in_axes=(0, None))
        vv_dist = jax.vmap(v_dist, in_axes=(None, 0))

        # Matrix of distances between all genotypes
        distances = vv_dist(descriptors, descriptors)

        inf_mask = jnp.logical_or(
            jnp.tile(repertoire_empty.reshape(1, -1), (num_genotypes, 1)),
            jnp.tile(repertoire_empty.reshape(1, -1), (num_genotypes, 1)).T,
        )
        distances = jnp.where(inf_mask, +jnp.inf, distances)
        distances = jnp.where(
            jnp.eye(num_genotypes) == 1, 0, distances
        )  # set diagonal to 0

        # Calculate novelty scores
        closest_distances, _ = jax.vmap(jax.lax.top_k, in_axes=(0, None))(
            distances, self._num_nn + 1
        )
        closest_distances = jnp.where(
            jnp.isinf(closest_distances), 0, closest_distances
        )
        novelty_scores = jax.vmap(lambda x: jnp.sum(x) / self._num_nn)(
            closest_distances
        )

        nonempty_novelty_scores = novelty_scores[~repertoire_empty]
        novelty_scores = jnp.where(
            repertoire_empty, jnp.min(nonempty_novelty_scores), novelty_scores
        )

        # probability of selecting each genotype
        p = novelty_scores - jnp.min(novelty_scores)
        p = p / jnp.sum()

        # select parents
        random_key, subkey = jax.random.split(random_key)
        selected_parents = jax.tree_util.tree_map(
            lambda x: jax.random.choice(
                subkey, x, shape=(number_parents_to_select,), p=p
            ),
            repertoire_genotypes,
        )

        return selected_parents, emitter_state, random_key
