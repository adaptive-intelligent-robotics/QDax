"""This file contains the class to define the repertoire used in
the MONET algorithm."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype


class MONETRepertoire(MapElitesRepertoire):
    """Class for the repertoire of the MONET algorithm.

    Identical to a MAP-Elites repertoire (in MONET, the cells correspond to
    the tasks and the centroids are the task descriptors), except for the
    acceptance rule: a new solution replaces the elite of its cell when its
    fitness is greater than *or equal to* the fitness of the elite
    ("bigger or equal" copy criterion in the reference implementation of
    MONET), which allows neutral genetic drift between tasks.
    """

    def add(  # type: ignore
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> MONETRepertoire:
        """
        Add a batch of elements to the repertoire. A new element replaces
        the current elite of its cell if its fitness is greater than or
        equal to the fitness of the elite.

        Args:
            batch_of_genotypes: a batch of genotypes to be added to the
                repertoire. Similarly to the self.genotypes argument, this
                is a PyTree in which the leaves have a shape
                (batch_size, num_features)
            batch_of_descriptors: an array that contains the descriptors of
                the aforementioned genotypes. Its shape is
                (batch_size, num_descriptors)
            batch_of_fitnesses: an array that contains the fitnesses of the
                aforementioned genotypes. Its shape is (batch_size,)
            batch_of_extra_scores: tree that contains the extra_scores of
                aforementioned genotypes.

        Returns:
            The updated MONET repertoire.
        """
        if batch_of_extra_scores is None:
            batch_of_extra_scores = {}

        filtered_batch_of_extra_scores = self.filter_extra_scores(batch_of_extra_scores)

        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)

        num_centroids = self.centroids.shape[0]
        batch_of_fitnesses = jnp.reshape(
            batch_of_fitnesses, (batch_of_descriptors.shape[0], 1)
        )

        # get fitness segment max
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices.astype(jnp.int32).squeeze(axis=-1),
            num_segments=num_centroids,
        )

        cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0)

        # put dominated fitness to -jnp.inf
        batch_of_fitnesses = jnp.where(
            batch_of_fitnesses == cond_values, batch_of_fitnesses, -jnp.inf
        )

        # get addition condition: greater or equal, restricted to
        # non-dominated candidates
        current_fitnesses = jnp.take_along_axis(self.fitnesses, batch_of_indices, 0)
        addition_condition = (batch_of_fitnesses >= current_fitnesses) & (
            batch_of_fitnesses > -jnp.inf
        )

        # assign fake position when relevant : num_centroids is out of bound
        batch_of_indices = jnp.where(
            addition_condition, batch_of_indices, num_centroids
        )

        # create new repertoire
        new_repertoire_genotypes = jax.tree.map(
            lambda repertoire_genotypes, new_genotypes: repertoire_genotypes.at[
                batch_of_indices.squeeze(axis=-1)
            ].set(new_genotypes),
            self.genotypes,
            batch_of_genotypes,
        )

        # compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_fitnesses
        )
        new_descriptors = self.descriptors.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_descriptors
        )

        # update extra scores
        new_extra_scores = jax.tree.map(
            lambda repertoire_scores, new_scores: repertoire_scores.at[
                batch_of_indices.squeeze(axis=-1)
            ].set(new_scores),
            self.extra_scores,
            filtered_batch_of_extra_scores,
        )

        return self.replace(  # type: ignore
            genotypes=new_repertoire_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            extra_scores=new_extra_scores,
        )
