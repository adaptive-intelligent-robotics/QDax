"""Core components of the Multi-Task Multi-Behavior MAP-Elites algorithm."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import jax

from qdax.core.containers.mtmb_repertoire import MTMBRepertoire
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


class MTMBMAPElites:
    """Core elements of the Multi-Task Multi-Behavior MAP-Elites algorithm.

    Multi-Task Multi-Behavior MAP-Elites (Anne and Mouret, GECCO 2023
    companion, https://arxiv.org/abs/2305.01264) combines MAP-Elites for the
    search for diversity and Multi-task MAP-Elites for leveraging the
    similarity between tasks: given a set of tasks, it builds one archive of
    diverse and high-performing solutions per task, all the archives sharing
    the same tessellation of the behavior space.

    Each offspring is evaluated on a task selected uniformly at random. The
    evaluation returns both the fitness and the behavior descriptor of the
    offspring on this task; the offspring then only competes in the behavior
    cell of the archive of this task.

    Note: the behavior space is discretized with a fixed tessellation shared
    by all the tasks (the standard MAP-Elites container), whereas the paper
    appends behavior bins as they are discovered.

    Args:
        scoring_function: a function that takes a batch of genotypes and the
            descriptors of the tasks on which they must be evaluated, and
            computes their fitnesses and their behavior descriptors on these
            tasks.
        emitter: an emitter used to suggest offsprings given a repertoire.
            The default selection of the MTMB repertoire is task-first, as
            in the paper: a random task with elites, then a random elite of
            this task.
        metrics_function: a function that takes an MTMB repertoire and
            computes any useful metric to track its evolution.
        repertoire_init: a function to initialize the repertoire.
    """

    def __init__(
        self,
        scoring_function: Optional[
            Callable[
                [Genotype, Descriptor, RNGKey],
                Tuple[Fitness, Descriptor, ExtraScores],
            ]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MTMBRepertoire], Metrics],
        repertoire_init: Callable[..., MTMBRepertoire] = MTMBRepertoire.init,
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._repertoire_init = repertoire_init

    def init(
        self,
        genotypes: Genotype,
        centroids: Centroid,
        task_descriptors: Descriptor,
        key: RNGKey,
    ) -> Tuple[MTMBRepertoire, Optional[EmitterState], Metrics]:
        """
        Initialize a Multi-Task Multi-Behavior repertoire with an initial
        population of genotypes, each evaluated on a randomly selected task.
        Requires the definition of a tessellation of the behavior space that
        can be computed with any method such as CVT or Euclidean mapping.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tessellation of the behavior space, shared by all the
                tasks, of shape (num_centroids, num_descriptors)
            task_descriptors: descriptors of the tasks to solve, of shape
                (num_tasks, num_task_descriptors)
            key: a random key used for stochastic operations.

        Returns:
            An initialized MTMB repertoire with the initial state of the
            emitter and initial metrics.
        """
        if self._scoring_function is None:
            raise ValueError("Scoring function is not set.")

        num_tasks = task_descriptors.shape[0]
        batch_size = jax.tree.leaves(genotypes)[0].shape[0]

        # evaluate each initial genotype on a random task
        key, subkey = jax.random.split(key)
        task_indices = jax.random.randint(subkey, (batch_size,), 0, num_tasks)
        key, subkey = jax.random.split(key)
        fitnesses, descriptors, extra_scores = self._scoring_function(
            genotypes, task_descriptors[task_indices], subkey
        )

        return self.init_ask_tell(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            task_indices=task_indices,
            centroids=centroids,
            task_descriptors=task_descriptors,
            key=key,
            extra_scores=extra_scores,
        )

    def init_ask_tell(
        self,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        task_indices: jax.Array,
        centroids: Centroid,
        task_descriptors: Descriptor,
        key: RNGKey,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Tuple[MTMBRepertoire, Optional[EmitterState], Metrics]:
        """
        Initialize a Multi-Task Multi-Behavior repertoire with an initial
        population of genotypes and their evaluations.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            fitnesses: fitnesses of the initial genotypes on their tasks
            descriptors: behavior descriptors of the initial genotypes on
                their tasks
            task_indices: indices of the tasks on which the initial
                genotypes have been evaluated, of shape (batch_size,)
            centroids: tessellation of the behavior space, of shape
                (num_centroids, num_descriptors)
            task_descriptors: descriptors of the tasks to solve, of shape
                (num_tasks, num_task_descriptors)
            key: a random key used for stochastic operations.
            extra_scores: extra scores of the initial genotypes (optional)

        Returns:
            An initialized MTMB repertoire with the initial state of the
            emitter and initial metrics.
        """
        if extra_scores is None:
            extra_scores = {}

        # init the repertoire
        repertoire = self._repertoire_init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            task_indices=task_indices,
            centroids=centroids,
            task_descriptors=task_descriptors,
            extra_scores=extra_scores,
        )

        # get initial state of the emitter
        key, subkey = jax.random.split(key)
        emitter_state = self._emitter.init(
            key=subkey,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # calculate the initial metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics

    def ask(
        self,
        repertoire: MTMBRepertoire,
        emitter_state: Optional[EmitterState],
        key: RNGKey,
    ) -> Tuple[Genotype, jax.Array, ExtraScores]:
        """
        Ask the emitter to generate a new batch of genotypes and select,
        uniformly at random, the task on which each of them will be
        evaluated.

        Args:
            repertoire: the MTMB repertoire
            emitter_state: state of the emitter
            key: a jax PRNG random key

        Returns:
            a new batch of genotypes, the indices of the tasks on which they
            must be evaluated and extra information from the emitter
        """
        key, subkey = jax.random.split(key)
        genotypes, extra_info = self._emitter.emit(repertoire, emitter_state, subkey)

        num_tasks = repertoire.task_descriptors.shape[0]
        batch_size = jax.tree.leaves(genotypes)[0].shape[0]
        key, subkey = jax.random.split(key)
        task_indices = jax.random.randint(subkey, (batch_size,), 0, num_tasks)

        return genotypes, task_indices, extra_info

    def tell(
        self,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        task_indices: jax.Array,
        repertoire: MTMBRepertoire,
        emitter_state: Optional[EmitterState],
        extra_scores: Optional[ExtraScores] = None,
        extra_info: Optional[ExtraScores] = None,
    ) -> Tuple[MTMBRepertoire, Optional[EmitterState], Metrics]:
        """
        Add new genotypes to the repertoire and update the emitter state.

        Args:
            genotypes: new genotypes to add to the repertoire
            fitnesses: fitnesses of the new genotypes on their tasks
            descriptors: behavior descriptors of the new genotypes on their
                tasks
            task_indices: indices of the tasks on which the new genotypes
                have been evaluated
            repertoire: the MTMB repertoire
            emitter_state: state of the emitter
            extra_scores: extra scores of the new genotypes
            extra_info: extra information from the emitter
        """
        if extra_scores is None:
            extra_scores = {}
        if extra_info is None:
            extra_info = {}

        # add genotypes in the repertoire
        repertoire = repertoire.add(
            genotypes, task_indices, descriptors, fitnesses, extra_scores
        )

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores={**extra_scores, **extra_info},
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics

    def update(
        self,
        repertoire: MTMBRepertoire,
        emitter_state: Optional[EmitterState],
        key: RNGKey,
    ) -> Tuple[MTMBRepertoire, Optional[EmitterState], Metrics]:
        """
        Performs one iteration of the Multi-Task Multi-Behavior MAP-Elites
        algorithm.
        1. A batch of genotypes is sampled in the repertoire and the
            genotypes are copied.
        2. The copies are mutated and crossed-over.
        3. A task is selected uniformly at random for each offspring.
        4. The offsprings are scored on their tasks, which gives their
            fitnesses and their behavior descriptors, and added to the
            archives of their tasks.

        Args:
            repertoire: the MTMB repertoire
            emitter_state: state of the emitter
            key: a jax PRNG random key

        Returns:
            the updated MTMB repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
        """
        if self._scoring_function is None:
            raise ValueError("Scoring function is not set.")

        # generate offsprings and select their tasks
        key, subkey = jax.random.split(key)
        genotypes, task_indices, extra_info = self.ask(
            repertoire, emitter_state, subkey
        )

        # score the offsprings on the selected tasks
        key, subkey = jax.random.split(key)
        fitnesses, descriptors, extra_scores = self._scoring_function(
            genotypes, repertoire.task_descriptors[task_indices], subkey
        )

        return self.tell(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            task_indices=task_indices,
            repertoire=repertoire,
            emitter_state=emitter_state,
            extra_scores=extra_scores,
            extra_info=extra_info,
        )

    def scan_update(
        self,
        carry: Tuple[MTMBRepertoire, Optional[EmitterState], RNGKey],
        _: Any,
    ) -> Tuple[Tuple[MTMBRepertoire, Optional[EmitterState], RNGKey], Metrics]:
        """Rewrites the update function in a way that makes it compatible
        with the jax.lax.scan primitive.

        Args:
            carry: a tuple containing the repertoire, the emitter state and
                a random key.
            _: unused element, necessary to respect jax.lax.scan API.

        Returns:
            The updated repertoire and emitter state, with a new random key
            and metrics.
        """
        repertoire, emitter_state, key = carry
        key, subkey = jax.random.split(key)
        (
            repertoire,
            emitter_state,
            metrics,
        ) = self.update(
            repertoire,
            emitter_state,
            subkey,
        )

        return (repertoire, emitter_state, key), metrics
