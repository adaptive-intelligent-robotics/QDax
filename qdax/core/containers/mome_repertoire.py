"""This file contains the class to define the repertoire used to
store individuals in the Multi-Objective MAP-Elites algorithm as
well as several variants."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax.core.emitters.repertoire_selectors.mome_uniform_selector import (
    MOMERepertoireT,
    MOMEUniformSelector,
)
from qdax.core.emitters.repertoire_selectors.selector import Selector
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Mask,
    ParetoFront,
    RNGKey,
)
from qdax.utils.pareto_front import compute_masked_pareto_front


class MOMERepertoire(MapElitesRepertoire):
    """Class for the repertoire in Multi Objective Map Elites

    This class inherits from MAPElitesRepertoire. The stored data
    is the same: genotypes, fitnesses, descriptors, centroids.

    The shape of genotypes is (in the case where it's an array):
    (num_centroids, pareto_front_length, genotype_dim).
    When the genotypes is a PyTree, the two first dimensions are the same
    but the third will depend on the leafs.

    The shape of fitnesses is: (num_centroids, pareto_front_length, num_criteria)

    The shape of descriptors and centroids are:
    (num_centroids, num_descriptors, pareto_front_length).
    """

    @property
    def repertoire_capacity(self) -> int:
        """Returns the maximum number of solutions the repertoire can
        contain which corresponds to the number of cells times the
        maximum pareto front length.

        Returns:
            The repertoire capacity.
        """
        first_leaf = jax.tree.leaves(self.genotypes)[0]
        return int(first_leaf.shape[0] * first_leaf.shape[1])

    def select(
        self,
        key: RNGKey,
        num_samples: int,
        selector: Optional[Selector[MOMERepertoireT]] = None,
    ) -> MOMERepertoireT:
        if selector is None:
            selector = MOMEUniformSelector()
        repertoire = selector.select(self, key, num_samples)
        return repertoire

    def _update_masked_pareto_front(
        self,
        pareto_front_fitnesses: ParetoFront[Fitness],
        pareto_front_genotypes: ParetoFront[Genotype],
        pareto_front_descriptors: ParetoFront[Descriptor],
        pareto_front_extra_scores: ParetoFront[ExtraScores],
        mask: Mask,
        new_batch_of_fitnesses: Fitness,
        new_batch_of_genotypes: Genotype,
        new_batch_of_descriptors: Descriptor,
        new_batch_of_extra_scores: ExtraScores,
        new_mask: Mask,
    ) -> Tuple[
        ParetoFront[Fitness],
        ParetoFront[Genotype],
        ParetoFront[Descriptor],
        ParetoFront[ExtraScores],
        Mask,
    ]:
        """Takes a fixed size pareto front, its mask and new points to add.
        Returns updated front and mask.

        Args:
            pareto_front_fitnesses: fitness of the pareto front
            pareto_front_genotypes: corresponding genotypes
            pareto_front_descriptors: corresponding descriptors
            pareto_front_extra_scores: corresponding extra scores
            mask: mask of the front, to hide void parts
            new_batch_of_fitnesses: new batch of fitness that is considered
                to be added to the pareto front
            new_batch_of_genotypes: corresponding genotypes
            new_batch_of_descriptors: corresponding descriptors
            new_batch_of_extra_scores: corresponding extra scores
            new_mask: corresponding mask (no one is masked)

        Returns:
            The updated pareto front.
        """
        # get dimensions
        batch_size = new_batch_of_fitnesses.shape[0]
        num_criteria = new_batch_of_fitnesses.shape[1]

        pareto_front_len = pareto_front_fitnesses.shape[0]  # type: ignore

        descriptors_dim = new_batch_of_descriptors.shape[1]

        # gather all data
        cat_mask = jnp.concatenate([mask, new_mask], axis=-1)
        cat_fitnesses = jnp.concatenate(
            [pareto_front_fitnesses, new_batch_of_fitnesses], axis=0
        )
        cat_genotypes = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            pareto_front_genotypes,
            new_batch_of_genotypes,
        )
        cat_descriptors = jnp.concatenate(
            [pareto_front_descriptors, new_batch_of_descriptors], axis=0
        )
        cat_extra_scores = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            pareto_front_extra_scores,
            new_batch_of_extra_scores,
        )

        # get new front
        cat_bool_front = compute_masked_pareto_front(
            batch_of_criteria=cat_fitnesses, mask=cat_mask
        )

        # get corresponding indices
        indices = (
            jnp.arange(start=0, stop=pareto_front_len + batch_size) * cat_bool_front
        )
        indices = indices + ~cat_bool_front * (batch_size + pareto_front_len - 1)
        indices = jnp.sort(indices)

        # get new fitness, genotypes and descriptors
        new_front_fitness = jnp.take(cat_fitnesses, indices, axis=0)
        new_front_genotypes = jax.tree.map(
            lambda x: jnp.take(x, indices, axis=0), cat_genotypes
        )
        new_front_descriptors = jnp.take(cat_descriptors, indices, axis=0)
        new_front_extra_scores = jax.tree.map(
            lambda x: jnp.take(x, indices, axis=0), cat_extra_scores
        )

        # compute new mask
        num_front_elements = jnp.sum(cat_bool_front)
        new_mask_indices = jnp.arange(start=0, stop=batch_size + pareto_front_len)
        new_mask_indices = (num_front_elements - new_mask_indices) > 0

        new_mask = jnp.where(
            new_mask_indices,
            jnp.ones(shape=batch_size + pareto_front_len, dtype=bool),
            jnp.zeros(shape=batch_size + pareto_front_len, dtype=bool),
        )

        fitness_mask = jnp.repeat(
            jnp.expand_dims(new_mask, axis=-1), num_criteria, axis=-1
        )
        new_front_fitness = new_front_fitness * fitness_mask

        front_size = len(pareto_front_fitnesses)  # type: ignore
        new_front_fitness = new_front_fitness[:front_size, :]

        new_front_genotypes = jax.tree.map(
            lambda x: x * new_mask_indices[0], new_front_genotypes
        )
        new_front_genotypes = jax.tree.map(
            lambda x: x[:front_size], new_front_genotypes
        )

        descriptors_mask = jnp.repeat(
            jnp.expand_dims(new_mask, axis=-1), descriptors_dim, axis=-1
        )
        new_front_descriptors = new_front_descriptors * descriptors_mask
        new_front_descriptors = new_front_descriptors[:front_size, :]

        new_front_extra_scores = jax.tree.map(
            lambda x: x * new_mask_indices[0], new_front_extra_scores
        )
        new_front_extra_scores = jax.tree.map(
            lambda x: x[:front_size], new_front_extra_scores
        )

        new_mask = ~new_mask[:front_size]

        return (
            new_front_fitness,
            new_front_genotypes,
            new_front_descriptors,
            new_front_extra_scores,
            new_mask,
        )

    def add(  # type: ignore
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> MOMERepertoire:
        """Insert a batch of elements in the repertoire.

        Shape of the batch_of_genotypes (if an array):
        (batch_size, genotypes_dim)
        Shape of the batch_of_descriptors: (batch_size, num_descriptors)
        Shape of the batch_of_fitnesses: (batch_size, num_criteria)

        Args:
            batch_of_genotypes: a batch of genotypes that we are trying to
                insert into the repertoire.
            batch_of_descriptors: the descriptors of the genotypes we are
                trying to add to the repertoire.
            batch_of_fitnesses: the fitnesses of the genotypes we are trying
                to add to the repertoire.
            batch_of_extra_scores: unused tree that contains the extra_scores of
                aforementioned genotypes.

        Returns:
            The updated repertoire with potential new individuals.
        """
        if batch_of_extra_scores is None:
            batch_of_extra_scores = {}

        filtered_batch_of_extra_scores = self.filter_extra_scores(batch_of_extra_scores)

        # get the indices that corresponds to the descriptors in the repertoire
        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)

        def _add_one(
            carry: MOMERepertoire,
            data: Tuple[Genotype, Descriptor, Fitness, ExtraScores, jnp.ndarray],
        ) -> Tuple[MOMERepertoire, Any]:
            # unwrap data
            genotype, descriptors, fitness, extra_scores, index = data

            index = index.astype(jnp.int32)

            # get current repertoire cell data
            cell_genotype = jax.tree.map(lambda x: x[index][0], carry.genotypes)
            cell_fitness = carry.fitnesses[index][0]
            cell_descriptor = carry.descriptors[index][0]
            cell_extra_scores = jax.tree.map(lambda x: x[index][0], carry.extra_scores)
            cell_mask = jnp.any(cell_fitness == -jnp.inf, axis=-1)

            new_genotypes = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), genotype)

            # update pareto front
            (
                cell_fitness,
                cell_genotype,  # new pf for cell
                cell_descriptor,
                cell_extra_scores,
                cell_mask,
            ) = self._update_masked_pareto_front(
                pareto_front_fitnesses=cell_fitness,
                pareto_front_genotypes=cell_genotype,
                pareto_front_descriptors=cell_descriptor,
                pareto_front_extra_scores=cell_extra_scores,
                mask=cell_mask,
                new_batch_of_fitnesses=jnp.expand_dims(fitness, axis=0),
                new_batch_of_genotypes=new_genotypes,
                new_batch_of_descriptors=jnp.expand_dims(descriptors, axis=0),
                new_batch_of_extra_scores=jax.tree.map(
                    lambda x: jnp.expand_dims(x, axis=0), extra_scores
                ),
                new_mask=jnp.zeros(shape=(1,), dtype=bool),
            )

            # update cell fitness
            cell_fitness = cell_fitness - jnp.inf * jnp.expand_dims(cell_mask, axis=-1)

            # update grid
            new_genotypes = jax.tree.map(
                lambda x, y: x.at[index].set(y), carry.genotypes, cell_genotype
            )
            new_fitnesses = carry.fitnesses.at[index].set(cell_fitness)
            new_descriptors = carry.descriptors.at[index].set(cell_descriptor)
            new_extra_scores = jax.tree.map(
                lambda x, y: x.at[index].set(y), carry.extra_scores, cell_extra_scores
            )
            carry = carry.replace(  # type: ignore
                genotypes=new_genotypes,
                descriptors=new_descriptors,
                fitnesses=new_fitnesses,
                extra_scores=new_extra_scores,
            )

            # return new grid
            return carry, ()

        # scan the addition operation for all the data
        self, _ = jax.lax.scan(
            _add_one,
            self,
            (
                batch_of_genotypes,
                batch_of_descriptors,
                batch_of_fitnesses,
                filtered_batch_of_extra_scores,
                batch_of_indices,
            ),
        )

        return self

    @classmethod
    def init(  # type: ignore
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        pareto_front_max_length: int,
        *args,
        extra_scores: Optional[ExtraScores] = None,
        keys_extra_scores: Tuple[str, ...] = (),
        **kwargs,
    ) -> MOMERepertoire:
        """
        Initialize a Multi Objective Map-Elites repertoire with an initial population
        of genotypes. Requires the definition of centroids that can be computed with
        any method such as CVT or Euclidean mapping.

        Note: this function has been kept outside of the object MapElites, so it can
        be called easily called from other modules.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            fitnesses: fitness of the initial genotypes of shape:
                (batch_size, num_criteria)
            descriptors: descriptors of the initial genotypes
                of shape (batch_size, num_descriptors)
            centroids: tessellation centroids of shape (batch_size, num_descriptors)
            pareto_front_max_length: maximum size of the pareto fronts
            extra_scores: unused extra_scores of the initial genotypes
            keys_extra_scores: keys of the extra_scores of the initial genotypes

        Returns:
            An initialized MAP-Elite repertoire
        """

        warnings.warn(
            (
                "This type of repertoire does not store the extra scores "
                "computed by the scoring function"
            ),
            stacklevel=2,
        )

        if extra_scores is None:
            extra_scores = {}

        filtered_extra_scores = {key: extra_scores[key] for key in keys_extra_scores}

        # get dimensions
        num_criteria = fitnesses.shape[1]
        num_descriptors = descriptors.shape[1]
        num_centroids = centroids.shape[0]

        # create default values
        default_fitnesses = -jnp.inf * jnp.ones(
            shape=(num_centroids, pareto_front_max_length, num_criteria)
        )
        default_genotypes = jax.tree.map(
            lambda x: jnp.zeros(
                shape=(
                    num_centroids,
                    pareto_front_max_length,
                )
                + x.shape[1:]
            ),
            genotypes,
        )
        default_descriptors = jnp.zeros(
            shape=(num_centroids, pareto_front_max_length, num_descriptors)
        )

        default_extra_scores = jax.tree.map(
            lambda x: jnp.zeros(
                shape=(num_centroids, pareto_front_max_length, *x.shape[1:])
            ),
            filtered_extra_scores,
        )

        # create repertoire with default values
        repertoire = MOMERepertoire(  # type: ignore
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
            extra_scores=default_extra_scores,
            keys_extra_scores=keys_extra_scores,
        )

        # add first batch of individuals in the repertoire
        new_repertoire = repertoire.add(genotypes, descriptors, fitnesses, extra_scores)

        return new_repertoire  # type: ignore

    @jax.jit
    def compute_global_pareto_front(
        self,
    ) -> Tuple[ParetoFront[Fitness], Mask]:
        """Merge all the pareto fronts of the MOME repertoire into a single one
        called global pareto front.

        Returns:
            The pareto front and its mask.
        """
        fitnesses = jnp.concatenate(self.fitnesses, axis=0)
        mask = jnp.any(fitnesses == -jnp.inf, axis=-1)
        pareto_mask = compute_masked_pareto_front(fitnesses, mask)
        pareto_front = fitnesses - jnp.inf * (~jnp.array([pareto_mask, pareto_mask]).T)

        return pareto_front, pareto_mask
