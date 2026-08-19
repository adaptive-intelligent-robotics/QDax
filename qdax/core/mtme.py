"""Core components of the Multi-task MAP-Elites algorithm."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import flax.struct
import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)


class MTMEState(flax.struct.PyTreeNode):
    """State of the UCB1 multi-armed bandit that controls the size of the
    tournament used to select the task on which each offspring is evaluated.

    Args:
        tournament_index: index (in the tuple of possible tournament sizes)
            of the tournament size used for the current generation.
        selection_counts: number of generations for which each tournament
            size has been selected.
        success_counts: number of offsprings added to the repertoire for
            each tournament size.
        generation: number of generations since the start of the algorithm.
    """

    tournament_index: jax.Array
    selection_counts: jax.Array
    success_counts: jax.Array
    generation: jax.Array


class MultiTaskMAPElites:
    """Core elements of the Multi-task MAP-Elites algorithm.

    Multi-task MAP-Elites (Mouret and Maguire, GECCO 2020,
    https://arxiv.org/abs/2003.04407) solves many tasks of the same family
    simultaneously when the fitness function depends on the task. Each cell
    of the repertoire corresponds to a task, described by a task descriptor
    (the task descriptors play the role of the centroids), and stores the
    best solution found so far for this task.

    The task on which a newly generated offspring is evaluated is selected
    with a tournament: a set of random tasks is sampled and the task closest
    (in task descriptor space) to the task of the first parent is kept. The
    size of the tournament is adjusted on the fly with a UCB1 multi-armed
    bandit that is rewarded by the number of offsprings added to the
    repertoire.

    Note: as in the reference implementation, the tournament tasks are
    sampled with replacement. The number of successes of a generation is
    computed by comparing each offspring to the elites stored in the
    repertoire before the whole batch is added, whereas the (sequential)
    reference implementation compares each offspring to the archive updated
    with the previous offsprings of the batch.

    Args:
        scoring_function: a function that takes a batch of genotypes and the
            descriptors of the tasks on which they must be evaluated, and
            computes their fitnesses.
        emitter: an emitter used to suggest offsprings given a repertoire.
            To use a tournament size greater than 1, the emitter must return
            the descriptors of the first parents in the extra information
            dictionary, under the key "parent_descriptors" (see
            MultiTaskMixingEmitter). With tournament_sizes=(1,), tasks are
            selected uniformly at random and any emitter can be used.
        metrics_function: a function that takes a repertoire and computes
            any useful metric to track its evolution.
        tournament_sizes: the possible tournament sizes among which the
            bandit chooses. Use (1,) to select tasks uniformly at random
            (the "random task" variant of the paper).
        repertoire_init: a function to initialize the repertoire.
    """

    def __init__(
        self,
        scoring_function: Optional[
            Callable[[Genotype, Descriptor, RNGKey], Tuple[Fitness, ExtraScores]]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
        tournament_sizes: Tuple[int, ...] = (1, 10, 100, 1000),
        repertoire_init: Callable[
            [Genotype, Fitness, Descriptor, Centroid, Optional[ExtraScores]],
            MapElitesRepertoire,
        ] = MapElitesRepertoire.init,
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._tournament_sizes = tuple(int(size) for size in tournament_sizes)
        self._repertoire_init = repertoire_init

    def init(
        self,
        genotypes: Genotype,
        task_descriptors: Descriptor,
        key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], MTMEState, Metrics]:
        """
        Initialize a Multi-task MAP-Elites repertoire with an initial
        population of genotypes, each evaluated on a randomly selected task.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            task_descriptors: descriptors of the tasks to solve, of shape
                (num_tasks, num_task_descriptors). They are used as the
                centroids of the repertoire.
            key: a random key used for stochastic operations.

        Returns:
            An initialized repertoire, the initial state of the emitter, the
            initial state of the task-selection bandit and initial metrics.
        """
        if self._scoring_function is None:
            raise ValueError("Scoring function is not set.")

        num_tasks = task_descriptors.shape[0]
        batch_size = jax.tree.leaves(genotypes)[0].shape[0]

        # evaluate each initial genotype on a random task
        key, subkey = jax.random.split(key)
        task_indices = jax.random.randint(subkey, (batch_size,), 0, num_tasks)
        key, subkey = jax.random.split(key)
        fitnesses, extra_scores = self._scoring_function(
            genotypes, task_descriptors[task_indices], subkey
        )

        return self.init_ask_tell(
            genotypes=genotypes,
            fitnesses=fitnesses,
            task_indices=task_indices,
            task_descriptors=task_descriptors,
            key=key,
            extra_scores=extra_scores,
        )

    def init_ask_tell(
        self,
        genotypes: Genotype,
        fitnesses: Fitness,
        task_indices: jax.Array,
        task_descriptors: Descriptor,
        key: RNGKey,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], MTMEState, Metrics]:
        """
        Initialize a Multi-task MAP-Elites repertoire with an initial
        population of genotypes and their evaluations.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            fitnesses: fitnesses of the initial genotypes, evaluated on the
                tasks given by task_indices
            task_indices: indices of the tasks on which the initial genotypes
                have been evaluated, of shape (batch_size,)
            task_descriptors: descriptors of the tasks to solve, of shape
                (num_tasks, num_task_descriptors)
            key: a random key used for stochastic operations.
            extra_scores: extra scores of the initial genotypes (optional)

        Returns:
            An initialized repertoire, the initial state of the emitter, the
            initial state of the task-selection bandit and initial metrics.
        """
        if extra_scores is None:
            extra_scores = {}

        # init the repertoire: the task descriptors are the centroids
        repertoire = self._repertoire_init(
            genotypes,
            fitnesses,
            task_descriptors[task_indices],
            task_descriptors,
            extra_scores,
        )

        # get initial state of the emitter
        key, subkey = jax.random.split(key)
        emitter_state = self._emitter.init(
            key=subkey,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=task_descriptors[task_indices],
            extra_scores=extra_scores,
        )

        # init the bandit state with a random tournament size
        num_arms = len(self._tournament_sizes)
        key, subkey = jax.random.split(key)
        mtme_state = MTMEState(
            tournament_index=jax.random.randint(subkey, (), 0, num_arms),
            selection_counts=jnp.zeros(num_arms),
            success_counts=jnp.zeros(num_arms),
            generation=jnp.zeros((), dtype=jnp.int32),
        )

        # calculate the initial metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, mtme_state, metrics

    def _select_tasks(
        self,
        repertoire: MapElitesRepertoire,
        extra_info: ExtraScores,
        mtme_state: MTMEState,
        batch_size: int,
        key: RNGKey,
    ) -> jax.Array:
        """Select the task on which each offspring will be evaluated.

        A tournament of tasks is sampled for each offspring and the task
        closest to the task of its first parent is kept. A tournament of
        size 1 amounts to a uniform random task selection.

        Args:
            repertoire: the repertoire, whose centroids are the task
                descriptors
            extra_info: extra information returned by the emitter, expected
                to contain the key "parent_descriptors" when a tournament
                size greater than 1 is used
            mtme_state: state of the task-selection bandit
            batch_size: number of offsprings
            key: a jax PRNG random key

        Returns:
            the indices of the selected tasks, of shape (batch_size,)
        """
        num_tasks = repertoire.centroids.shape[0]
        max_size = max(self._tournament_sizes)

        if "parent_descriptors" not in extra_info:
            if max_size > 1:
                raise ValueError(
                    "The emitter did not provide 'parent_descriptors' in its "
                    "extra information, which is required to run the task "
                    "selection tournament. Use an emitter that returns the "
                    "parent descriptors (e.g. MultiTaskMixingEmitter) or set "
                    "tournament_sizes=(1,) for uniform task selection."
                )
            return jax.random.randint(key, (batch_size,), 0, num_tasks)

        parent_descriptors = extra_info["parent_descriptors"]

        # sample the largest tournament and mask the extra candidates so that
        # the shapes remain static under jit
        tournament_size = jnp.array(self._tournament_sizes)[mtme_state.tournament_index]
        candidates = jax.random.randint(key, (batch_size, max_size), 0, num_tasks)
        distances = jnp.sum(
            jnp.square(
                repertoire.centroids[candidates] - parent_descriptors[:, None, :]
            ),
            axis=-1,
        )
        in_tournament = jnp.arange(max_size)[None, :] < tournament_size
        distances = jnp.where(in_tournament, distances, jnp.inf)
        return candidates[jnp.arange(batch_size), jnp.argmin(distances, axis=1)]

    def ask(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        mtme_state: MTMEState,
        key: RNGKey,
    ) -> Tuple[Genotype, jax.Array, ExtraScores]:
        """
        Ask the emitter to generate a new batch of genotypes and select the
        task on which each of them will be evaluated.

        Args:
            repertoire: the Multi-task MAP-Elites repertoire
            emitter_state: state of the emitter
            mtme_state: state of the task-selection bandit
            key: a jax PRNG random key

        Returns:
            a new batch of genotypes, the indices of the tasks on which they
            must be evaluated and extra information from the emitter
        """
        key, subkey = jax.random.split(key)
        genotypes, extra_info = self._emitter.emit(repertoire, emitter_state, subkey)

        batch_size = jax.tree.leaves(genotypes)[0].shape[0]
        key, subkey = jax.random.split(key)
        task_indices = self._select_tasks(
            repertoire, extra_info, mtme_state, batch_size, subkey
        )
        return genotypes, task_indices, extra_info

    def tell(
        self,
        genotypes: Genotype,
        fitnesses: Fitness,
        task_indices: jax.Array,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        mtme_state: MTMEState,
        extra_scores: Optional[ExtraScores] = None,
        extra_info: Optional[ExtraScores] = None,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], MTMEState, Metrics]:
        """
        Add new genotypes to the repertoire and update the states of the
        emitter and of the task-selection bandit.

        Args:
            genotypes: new genotypes to add to the repertoire
            fitnesses: fitnesses of the new genotypes on their tasks
            task_indices: indices of the tasks on which the new genotypes
                have been evaluated
            repertoire: the Multi-task MAP-Elites repertoire
            emitter_state: state of the emitter
            mtme_state: state of the task-selection bandit
            extra_scores: extra scores of the new genotypes
            extra_info: extra information from the emitter
        """
        if extra_scores is None:
            extra_scores = {}
        if extra_info is None:
            extra_info = {}

        fitnesses = jnp.ravel(fitnesses)
        task_descriptors = repertoire.centroids[task_indices]

        # count the offsprings that outperform the current elite of their task
        current_fitnesses = repertoire.fitnesses[task_indices, 0]
        num_successes = jnp.sum(fitnesses > current_fitnesses)

        # add genotypes to the repertoire: as the descriptor of an offspring
        # is exactly the descriptor of its task, it lands in the task's cell
        repertoire = repertoire.add(
            genotypes, task_descriptors, fitnesses, extra_scores
        )

        # update the bandit state and select the tournament size of the next
        # generation with UCB1
        arm = mtme_state.tournament_index
        selection_counts = mtme_state.selection_counts.at[arm].add(1.0)
        success_counts = mtme_state.success_counts.at[arm].add(num_successes)
        generation = mtme_state.generation + 1
        mean_rewards = success_counts / jnp.maximum(selection_counts, 1.0)
        exploration = jnp.sqrt(
            2.0
            * jnp.log(generation.astype(jnp.float32))
            / jnp.maximum(selection_counts, 1.0)
        )
        ucb_scores = jnp.where(
            selection_counts > 0.0, mean_rewards + exploration, jnp.inf
        )
        mtme_state = MTMEState(
            tournament_index=jnp.argmax(ucb_scores),
            selection_counts=selection_counts,
            success_counts=success_counts,
            generation=generation,
        )

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=task_descriptors,
            extra_scores={**extra_scores, **extra_info},
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, mtme_state, metrics

    def update(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        mtme_state: MTMEState,
        key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], MTMEState, Metrics]:
        """
        Performs one iteration of the Multi-task MAP-Elites algorithm.
        1. A batch of genotypes is sampled in the repertoire and the
            genotypes are copied.
        2. The copies are mutated and crossed-over.
        3. A task is selected for each offspring with a tournament whose
            size is chosen by a UCB1 bandit.
        4. The offsprings are scored on their tasks and added to the
            repertoire.

        Args:
            repertoire: the Multi-task MAP-Elites repertoire
            emitter_state: state of the emitter
            mtme_state: state of the task-selection bandit
            key: a jax PRNG random key

        Returns:
            the updated repertoire, the updated (if needed) emitter state,
            the updated bandit state and metrics about the repertoire
        """
        if self._scoring_function is None:
            raise ValueError("Scoring function is not set.")

        # generate offsprings and select their tasks
        key, subkey = jax.random.split(key)
        genotypes, task_indices, extra_info = self.ask(
            repertoire, emitter_state, mtme_state, subkey
        )

        # score the offsprings on the selected tasks
        key, subkey = jax.random.split(key)
        fitnesses, extra_scores = self._scoring_function(
            genotypes, repertoire.centroids[task_indices], subkey
        )

        return self.tell(
            genotypes=genotypes,
            fitnesses=fitnesses,
            task_indices=task_indices,
            repertoire=repertoire,
            emitter_state=emitter_state,
            mtme_state=mtme_state,
            extra_scores=extra_scores,
            extra_info=extra_info,
        )

    def scan_update(
        self,
        carry: Tuple[MapElitesRepertoire, Optional[EmitterState], MTMEState, RNGKey],
        _: Any,
    ) -> Tuple[
        Tuple[MapElitesRepertoire, Optional[EmitterState], MTMEState, RNGKey],
        Metrics,
    ]:
        """Rewrites the update function in a way that makes it compatible
        with the jax.lax.scan primitive.

        Args:
            carry: a tuple containing the repertoire, the emitter state, the
                bandit state and a random key.
            _: unused element, necessary to respect jax.lax.scan API.

        Returns:
            The updated repertoire, emitter state and bandit state, with a
            new random key and metrics.
        """
        repertoire, emitter_state, mtme_state, key = carry
        key, subkey = jax.random.split(key)
        (
            repertoire,
            emitter_state,
            mtme_state,
            metrics,
        ) = self.update(
            repertoire,
            emitter_state,
            mtme_state,
            subkey,
        )

        return (repertoire, emitter_state, mtme_state, key), metrics
