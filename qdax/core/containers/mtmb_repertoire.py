"""This file contains the class to define the repertoire used in
the Multi-Task Multi-Behavior MAP-Elites algorithm."""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax.core.emitters.repertoire_selectors.selector import (
    MapElitesRepertoireT,
    Selector,
)
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    RNGKey,
)


class MTMBRepertoire(MapElitesRepertoire):
    """Class for the repertoire of the Multi-Task Multi-Behavior MAP-Elites
    algorithm.

    The repertoire stores one full MAP-Elites archive per task, all the
    tasks sharing the same tessellation of the behavior space. The storage
    is flattened: the elite of the behavior cell c of the task t is stored
    at position t * num_centroids + c.

    Args:
        genotypes: a PyTree containing all the genotypes in the repertoire.
            Each leaf has a shape (num_tasks * num_centroids, num_features).
        fitnesses: fitness of the solution in each cell of the repertoire,
            of shape (num_tasks * num_centroids, 1).
        descriptors: behavior descriptors of the solutions in each cell of
            the repertoire, of shape
            (num_tasks * num_centroids, num_descriptors).
        centroids: tessellation of the behavior space, shared by all the
            tasks, of shape (num_centroids, num_descriptors).
        task_descriptors: descriptors of the tasks, of shape
            (num_tasks, num_task_descriptors).
        extra_scores: extra scores resulting from the evaluation of the
            genotypes.
        keys_extra_scores: keys of the extra scores to store in the
            repertoire.
    """

    task_descriptors: Descriptor

    @property
    def num_tasks(self) -> int:
        """Gives the number of tasks of the repertoire."""
        return int(self.task_descriptors.shape[0])

    @property
    def num_centroids_per_task(self) -> int:
        """Gives the number of behavior cells of each task."""
        return int(self.centroids.shape[0])

    def select(
        self,
        key: RNGKey,
        num_samples: int,
        selector: Optional[Selector[MapElitesRepertoireT]] = None,
    ) -> MTMBRepertoire:
        """Selects individuals from the repertoire.

        By default, the selection is task-first, as in the Multi-Task
        Multi-Behavior MAP-Elites paper: a task is selected uniformly at
        random among the tasks that contain at least one elite, then an
        elite is selected uniformly at random within this task. Note that
        generic selectors relying on unfold_repertoire are not compatible
        with this repertoire.

        Args:
            key: The random key to use for the selection.
            num_samples: The number of individuals to select.
            selector: An optional custom selector.

        Returns:
            A repertoire containing the selected individuals.
        """
        if selector is not None:
            return selector.select(self, key, num_samples)  # type: ignore

        num_centroids = self.centroids.shape[0]
        num_tasks = self.task_descriptors.shape[0]

        occupancy = (self.fitnesses[:, 0] != -jnp.inf).reshape(num_tasks, num_centroids)

        # select tasks uniformly among the tasks with at least one elite
        task_has_elite = jnp.any(occupancy, axis=1)
        p_task = task_has_elite / jnp.sum(task_has_elite)
        key, subkey = jax.random.split(key)
        task_indices = jax.random.choice(
            subkey, num_tasks, shape=(num_samples,), p=p_task
        )

        # select elites uniformly within each selected task
        cell_logits = jnp.where(occupancy[task_indices], 0.0, -jnp.inf)
        keys = jax.random.split(key, num_samples)
        cell_indices = jax.vmap(jax.random.categorical)(keys, cell_logits)

        selected_indices = task_indices * num_centroids + cell_indices

        selected: MTMBRepertoire = self.replace(  # type: ignore
            genotypes=jax.tree.map(lambda x: x[selected_indices], self.genotypes),
            fitnesses=self.fitnesses[selected_indices],
            descriptors=self.descriptors[selected_indices],
            extra_scores=jax.tree.map(lambda x: x[selected_indices], self.extra_scores),
            centroids=self.centroids[cell_indices],
            task_descriptors=self.task_descriptors[task_indices],
        )
        return selected

    def add(  # type: ignore
        self,
        batch_of_genotypes: Genotype,
        batch_of_task_indices: jax.Array,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> MTMBRepertoire:
        """
        Add a batch of elements to the repertoire. Each element only
        competes with the elites of the task on which it has been evaluated,
        in the behavior cell corresponding to its descriptor.

        Args:
            batch_of_genotypes: a batch of genotypes to be added to the
                repertoire. Similarly to the self.genotypes argument, this
                is a PyTree in which the leaves have a shape
                (batch_size, num_features)
            batch_of_task_indices: an array that contains the indices of the
                tasks on which the aforementioned genotypes have been
                evaluated. Its shape is (batch_size,)
            batch_of_descriptors: an array that contains the behavior
                descriptors of the aforementioned genotypes. Its shape is
                (batch_size, num_descriptors)
            batch_of_fitnesses: an array that contains the fitnesses of the
                aforementioned genotypes. Its shape is (batch_size,)
            batch_of_extra_scores: tree that contains the extra_scores of
                aforementioned genotypes.

        Returns:
            The updated MTMB repertoire.
        """
        if batch_of_extra_scores is None:
            batch_of_extra_scores = {}

        filtered_batch_of_extra_scores = self.filter_extra_scores(batch_of_extra_scores)

        num_centroids = self.centroids.shape[0]
        num_cells = self.fitnesses.shape[0]

        behavior_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        batch_of_indices = batch_of_task_indices * num_centroids + behavior_indices
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)

        batch_of_fitnesses = jnp.reshape(
            batch_of_fitnesses, (batch_of_descriptors.shape[0], 1)
        )

        # get fitness segment max
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices.astype(jnp.int32).squeeze(axis=-1),
            num_segments=num_cells,
        )

        cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0)

        # put dominated fitness to -jnp.inf
        batch_of_fitnesses = jnp.where(
            batch_of_fitnesses == cond_values, batch_of_fitnesses, -jnp.inf
        )

        # get addition condition
        current_fitnesses = jnp.take_along_axis(self.fitnesses, batch_of_indices, 0)
        addition_condition = batch_of_fitnesses > current_fitnesses

        # assign fake position when relevant : num_cells is out of bound
        batch_of_indices = jnp.where(addition_condition, batch_of_indices, num_cells)

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

    def get_task_repertoire(self, task_index: int) -> MapElitesRepertoire:
        """Extract the MAP-Elites repertoire of a single task.

        Useful to reuse the tools built for MAP-Elites repertoires, such as
        the plotting utilities.

        Args:
            task_index: the index of the task.

        Returns:
            The MAP-Elites repertoire of the task.
        """
        num_centroids = self.centroids.shape[0]
        start = task_index * num_centroids

        def _slice(x: jax.Array) -> jax.Array:
            return jax.lax.dynamic_slice_in_dim(x, start, num_centroids, axis=0)

        return MapElitesRepertoire(
            genotypes=jax.tree.map(_slice, self.genotypes),
            fitnesses=_slice(self.fitnesses),
            descriptors=_slice(self.descriptors),
            centroids=self.centroids,
            extra_scores=jax.tree.map(_slice, self.extra_scores),
            keys_extra_scores=self.keys_extra_scores,
        )

    @classmethod
    def init(  # type: ignore
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        task_indices: jax.Array,
        centroids: Centroid,
        task_descriptors: Descriptor,
        extra_scores: Optional[ExtraScores] = None,
        keys_extra_scores: Tuple[str, ...] = (),
    ) -> MTMBRepertoire:
        """
        Initialize a Multi-Task Multi-Behavior repertoire with an initial
        population of genotypes. Requires the definition of a tessellation
        of the behavior space (shared by all the tasks) and of the task
        descriptors.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            fitnesses: fitness of the initial genotypes of shape
                (batch_size,)
            descriptors: behavior descriptors of the initial genotypes
                of shape (batch_size, num_descriptors)
            task_indices: indices of the tasks on which the initial
                genotypes have been evaluated, of shape (batch_size,)
            centroids: tessellation of the behavior space, of shape
                (num_centroids, num_descriptors)
            task_descriptors: descriptors of the tasks, of shape
                (num_tasks, num_task_descriptors)
            extra_scores: extra scores of the initial genotypes
            keys_extra_scores: keys of the extra scores to store in the
                repertoire

        Returns:
            an initialized MTMB repertoire
        """
        if extra_scores is None:
            extra_scores = {}

        extra_scores = {
            key: value
            for key, value in extra_scores.items()
            if key in keys_extra_scores
        }

        # retrieve one genotype from the population
        first_genotype = jax.tree.map(lambda x: x[0], genotypes)
        first_extra_scores = jax.tree.map(lambda x: x[0], extra_scores)

        # create a repertoire with default values
        repertoire = cls.init_default(
            genotype=first_genotype,
            centroids=centroids,
            task_descriptors=task_descriptors,
            one_extra_score=first_extra_scores,
            keys_extra_scores=keys_extra_scores,
        )

        # add initial population to the repertoire
        new_repertoire = repertoire.add(
            genotypes, task_indices, descriptors, fitnesses, extra_scores
        )

        return new_repertoire

    @classmethod
    def init_default(  # type: ignore
        cls,
        genotype: Genotype,
        centroids: Centroid,
        task_descriptors: Descriptor,
        one_extra_score: Optional[ExtraScores] = None,
        keys_extra_scores: Tuple[str, ...] = (),
    ) -> MTMBRepertoire:
        """Initialize a Multi-Task Multi-Behavior repertoire filled with
        default values.

        Args:
            genotype: the typical genotype that will be stored.
            centroids: tessellation of the behavior space.
            task_descriptors: descriptors of the tasks.
            one_extra_score: the typical extra score that will be stored.
            keys_extra_scores: keys of the extra scores to store in the
                repertoire

        Returns:
            A repertoire filled with default values.
        """
        if one_extra_score is None:
            one_extra_score = {}

        one_extra_score = {
            key: value
            for key, value in one_extra_score.items()
            if key in keys_extra_scores
        }

        # get total number of cells
        num_cells = task_descriptors.shape[0] * centroids.shape[0]

        # default fitness is -inf
        default_fitnesses = -jnp.inf * jnp.ones(shape=(num_cells, 1))

        # default genotypes is all 0
        default_genotypes = jax.tree.map(
            lambda x: jnp.zeros(shape=(num_cells,) + x.shape, dtype=x.dtype),
            genotype,
        )

        # default descriptor is all zeros
        default_descriptors = jnp.zeros(shape=(num_cells, centroids.shape[-1]))

        # default extra scores is empty dict
        default_extra_scores = jax.tree.map(
            lambda x: jnp.zeros(shape=(num_cells,) + x.shape, dtype=x.dtype),
            one_extra_score,
        )

        return cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
            task_descriptors=task_descriptors,
            extra_scores=default_extra_scores,
            keys_extra_scores=keys_extra_scores,
        )
