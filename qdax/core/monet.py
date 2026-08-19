"""Core components of the MONET algorithm."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import flax.struct
import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.containers.monet_repertoire import MONETRepertoire
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)


class MONETState(flax.struct.PyTreeNode):
    """State of the MONET algorithm.

    Args:
        neighbor_indices: for each task, the indices of its neighbors in the
            task similarity graph, sorted by decreasing similarity, of shape
            (num_tasks, num_neighbors). This is static: the graph is built
            once from the task descriptors at initialization.
    """

    neighbor_indices: jax.Array


def _tree_where(mask: jax.Array, on_true: Genotype, on_false: Genotype) -> Genotype:
    """Select between two pytrees element-wise with a (batch_size,) mask."""
    return jax.tree.map(
        lambda a, b: jnp.where(
            mask.reshape(mask.shape + (1,) * (a.ndim - mask.ndim)), a, b
        ),
        on_true,
        on_false,
    )


def _tree_uniform(
    like: Genotype, key: RNGKey, minval: float, maxval: float
) -> Genotype:
    """Sample a pytree of uniform random values with the same shapes as like."""
    leaves, treedef = jax.tree.flatten(like)
    keys = jax.random.split(key, len(leaves))
    new_leaves = [
        jax.random.uniform(k, leaf.shape, minval=minval, maxval=maxval)
        for k, leaf in zip(keys, leaves)
    ]
    return jax.tree.unflatten(treedef, new_leaves)


def _gaussian_mutation(
    x: Genotype, key: RNGKey, mutation_std: float, minval: float, maxval: float
) -> Genotype:
    """Gaussian mutation, with a standard deviation scaled by the range of
    the solution space, as in the reference implementation of MONET."""
    leaves, treedef = jax.tree.flatten(x)
    keys = jax.random.split(key, len(leaves))
    new_leaves = [
        jnp.clip(
            leaf + jax.random.normal(k, leaf.shape) * mutation_std * (maxval - minval),
            minval,
            maxval,
        )
        for k, leaf in zip(keys, leaves)
    ]
    return jax.tree.unflatten(treedef, new_leaves)


def _polynomial_mutation(
    x: Genotype, key: RNGKey, eta: float, minval: float, maxval: float
) -> Genotype:
    """Polynomial mutation (cf Deb 2001, p 124), as implemented in the
    reference implementation of MONET."""
    leaves, treedef = jax.tree.flatten(x)
    keys = jax.random.split(key, len(leaves))

    def _mutate(leaf: jax.Array, k: RNGKey) -> jax.Array:
        r = jax.random.uniform(k, leaf.shape)
        delta = jnp.where(
            r < 0.5,
            jnp.power(2.0 * r, 1.0 / (eta + 1.0)) - 1.0,
            1.0 - jnp.power(2.0 * (1.0 - r), 1.0 / (eta + 1.0)),
        )
        return jnp.clip(leaf + delta, minval, maxval)

    return jax.tree.unflatten(treedef, [_mutate(le, k) for le, k in zip(leaves, keys)])


def _sbx_crossover(
    x: Genotype, y: Genotype, key: RNGKey, eta: float, minval: float, maxval: float
) -> Genotype:
    """Simulated Binary Crossover (cf Deb 2001, p 113), as implemented in
    the reference implementation of MONET."""
    x_leaves, treedef = jax.tree.flatten(x)
    y_leaves = jax.tree.leaves(y)
    keys = jax.random.split(key, len(x_leaves))

    def _cross(x_leaf: jax.Array, y_leaf: jax.Array, k: RNGKey) -> jax.Array:
        k_1, k_2 = jax.random.split(k)
        r_1 = jax.random.uniform(k_1, x_leaf.shape)
        r_2 = jax.random.uniform(k_2, x_leaf.shape)

        x_low = jnp.minimum(x_leaf, y_leaf)
        x_high = jnp.maximum(x_leaf, y_leaf)
        delta = x_high - x_low
        valid = delta > 1e-15
        safe_delta = jnp.where(valid, delta, 1.0)

        def _beta_q(beta: jax.Array) -> jax.Array:
            alpha = 2.0 - jnp.power(beta, -(eta + 1.0))
            return jnp.where(
                r_1 <= 1.0 / alpha,
                jnp.power(r_1 * alpha, 1.0 / (eta + 1.0)),
                jnp.power(1.0 / (2.0 - r_1 * alpha), 1.0 / (eta + 1.0)),
            )

        beta_q_low = _beta_q(1.0 + 2.0 * (x_low - minval) / safe_delta)
        c_1 = 0.5 * (x_low + x_high - beta_q_low * delta)
        beta_q_high = _beta_q(1.0 + 2.0 * (maxval - x_high) / safe_delta)
        c_2 = 0.5 * (x_low + x_high + beta_q_high * delta)

        c_1 = jnp.clip(c_1, minval, maxval)
        c_2 = jnp.clip(c_2, minval, maxval)
        z = jnp.where(r_2 <= 0.5, c_2, c_1)
        return jnp.where(valid, z, x_leaf)

    return jax.tree.unflatten(
        treedef, [_cross(xl, yl, k) for xl, yl, k in zip(x_leaves, y_leaves, keys)]
    )


def _iso_dd_crossover(
    x: Genotype,
    y: Genotype,
    key: RNGKey,
    iso_sigma: float,
    line_sigma: float,
    minval: float,
    maxval: float,
) -> Genotype:
    """Iso+Line variation (Vassiliades and Mouret, GECCO 2018), as
    implemented in the reference implementation of MONET."""
    x_leaves, treedef = jax.tree.flatten(x)
    y_leaves = jax.tree.leaves(y)
    batch_size = x_leaves[0].shape[0]

    key, subkey = jax.random.split(key)
    # one line coefficient per individual, shared across the genotype
    line = jax.random.normal(subkey, (batch_size,)) * line_sigma

    keys = jax.random.split(key, len(x_leaves))
    new_leaves = []
    for x_leaf, y_leaf, k in zip(x_leaves, y_leaves, keys):
        iso = jax.random.normal(k, x_leaf.shape) * iso_sigma
        line_broadcast = line.reshape((batch_size,) + (1,) * (x_leaf.ndim - 1))
        z = x_leaf + iso + line_broadcast * (x_leaf - y_leaf)
        new_leaves.append(jnp.clip(z, minval, maxval))
    return jax.tree.unflatten(treedef, new_leaves)


class MONET:
    """Core elements of the MONET algorithm.

    MONET (Multi-Task Optimization over Networks of Tasks, Hatzky et al.,
    https://arxiv.org/abs/2604.21991) is a graph-based multi-task
    optimization algorithm: the tasks are the nodes of a similarity graph
    and each node stores the best solution found so far for its task (as in
    Multi-task MAP-Elites, the task descriptors play the role of the
    centroids of the repertoire). At every step, a random focal task is
    selected and, with probability p_ind, its elite is mutated (individual
    learning); otherwise, its elite is crossed-over with the elite of one of
    its nearest neighbors in the graph (social learning). The offspring is
    evaluated on the focal task and replaces the elite if its fitness is
    greater than or equal to the fitness of the elite.

    This implementation processes offsprings in batches, mirroring the
    parallel mode of the reference implementation
    (https://github.com/ju2ez/MONET): a batch of candidates is prepared from
    a snapshot of the archive, evaluated in parallel and committed with
    elitism. As in the reference implementation, when the selected neighbor
    has no elite yet, social learning falls back to a Gaussian mutation of
    the focal elite; and when the focal task has no elite yet, the offspring
    is a copy of the neighbor's elite (init_strategy="copy") or a random
    solution.

    Note: the "regression" and "plane_dd" operators and the random /
    distance-proportional neighborhood modes of the reference implementation
    are not supported.

    Args:
        scoring_function: a function that takes a batch of genotypes and the
            descriptors of the tasks on which they must be evaluated, and
            computes their fitnesses.
        metrics_function: a function that takes a repertoire and computes
            any useful metric to track its evolution.
        batch_size: number of offsprings generated and evaluated at each
            update.
        num_neighbors: number of nearest neighbors of each task in the task
            similarity graph (the reference implementation uses 1% of the
            number of tasks).
        p_ind: probability of individual learning (mutation); social
            learning (crossover with a neighbor) is applied otherwise.
        neighbor_strategy: how the neighbor used for social learning is
            selected among the neighbors of the focal task, one of
            "best_fitness", "random", "most_similar", "least_similar" or
            "fitness_proportional".
        individual_learning: the mutation operators used for individual
            learning, a tuple with elements among "gaussian_mutation" and
            "polynomial_mutation". If several operators are given, one is
            selected uniformly at random for each offspring.
        social_learning: the crossover operators used for social learning, a
            tuple with elements among "sbx" and "iso_dd". If several
            operators are given, one is selected uniformly at random for
            each offspring.
        init_strategy: how the offspring of a focal task without elite is
            initialized: "copy" copies the elite of the selected neighbor
            (when social learning is applied and the neighbor has an elite),
            "random" samples a random solution.
        mutation_std: standard deviation of the Gaussian mutation, relative
            to the range of the solution space.
        sbx_eta: eta parameter of the SBX crossover.
        polynomial_eta: eta parameter of the polynomial mutation.
        iso_sigma: iso parameter of the Iso+Line variation.
        line_sigma: line parameter of the Iso+Line variation.
        top_k_similar: number of candidate neighbors considered by the
            "most_similar" and "least_similar" strategies.
        minval: minimum value of the solution space.
        maxval: maximum value of the solution space.
        repertoire_init: a function to initialize the repertoire.
    """

    individual_learning_operators = ("gaussian_mutation", "polynomial_mutation")
    social_learning_operators = ("sbx", "iso_dd")
    neighbor_strategies = (
        "best_fitness",
        "random",
        "most_similar",
        "least_similar",
        "fitness_proportional",
    )

    def __init__(
        self,
        scoring_function: Optional[
            Callable[[Genotype, Descriptor, RNGKey], Tuple[Fitness, ExtraScores]]
        ],
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
        batch_size: int,
        num_neighbors: int,
        p_ind: float = 0.5,
        neighbor_strategy: str = "best_fitness",
        individual_learning: Tuple[str, ...] = ("gaussian_mutation",),
        social_learning: Tuple[str, ...] = ("sbx",),
        init_strategy: str = "copy",
        mutation_std: float = 0.05,
        sbx_eta: float = 10.0,
        polynomial_eta: float = 5.0,
        iso_sigma: float = 1.0 / 300.0,
        line_sigma: float = 20.0 / 300.0,
        top_k_similar: int = 3,
        minval: float = 0.0,
        maxval: float = 1.0,
        repertoire_init: Callable[
            [Genotype, Fitness, Descriptor, Centroid, Optional[ExtraScores]],
            MONETRepertoire,
        ] = MONETRepertoire.init,
    ) -> None:
        for operator in individual_learning:
            if operator not in self.individual_learning_operators:
                raise ValueError(
                    f"Unknown individual learning operator: {operator}. "
                    f"Supported operators: {self.individual_learning_operators}."
                )
        for operator in social_learning:
            if operator not in self.social_learning_operators:
                raise ValueError(
                    f"Unknown social learning operator: {operator}. "
                    f"Supported operators: {self.social_learning_operators}."
                )
        if neighbor_strategy not in self.neighbor_strategies:
            raise ValueError(
                f"Unknown neighbor selection strategy: {neighbor_strategy}. "
                f"Supported strategies: {self.neighbor_strategies}."
            )
        if init_strategy not in ("copy", "random"):
            raise ValueError(
                f"Unknown init strategy: {init_strategy}. "
                'Supported strategies: ("copy", "random").'
            )

        self._scoring_function = scoring_function
        self._metrics_function = metrics_function
        self._batch_size = batch_size
        self._num_neighbors = num_neighbors
        self._p_ind = p_ind
        self._neighbor_strategy = neighbor_strategy
        self._individual_learning_ops = tuple(individual_learning)
        self._social_learning_ops = tuple(social_learning)
        self._init_strategy = init_strategy
        self._mutation_std = mutation_std
        self._sbx_eta = sbx_eta
        self._polynomial_eta = polynomial_eta
        self._iso_sigma = iso_sigma
        self._line_sigma = line_sigma
        self._top_k_similar = top_k_similar
        self._minval = minval
        self._maxval = maxval
        self._repertoire_init = repertoire_init

    def init(
        self,
        genotypes: Genotype,
        task_descriptors: Descriptor,
        key: RNGKey,
    ) -> Tuple[MONETRepertoire, MONETState, Metrics]:
        """
        Initialize a MONET repertoire with an initial population of
        genotypes, each evaluated on a randomly selected task, and build the
        task similarity graph.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            task_descriptors: descriptors of the tasks to solve, of shape
                (num_tasks, num_task_descriptors). They are used as the
                centroids of the repertoire.
            key: a random key used for stochastic operations.

        Returns:
            An initialized repertoire, the MONET state (holding the task
            similarity graph) and initial metrics.
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
    ) -> Tuple[MONETRepertoire, MONETState, Metrics]:
        """
        Initialize a MONET repertoire with an initial population of
        genotypes and their evaluations, and build the task similarity
        graph.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            fitnesses: fitnesses of the initial genotypes, evaluated on the
                tasks given by task_indices
            task_indices: indices of the tasks on which the initial
                genotypes have been evaluated, of shape (batch_size,)
            task_descriptors: descriptors of the tasks to solve, of shape
                (num_tasks, num_task_descriptors)
            key: a random key used for stochastic operations.
            extra_scores: extra scores of the initial genotypes (optional)

        Returns:
            An initialized repertoire, the MONET state (holding the task
            similarity graph) and initial metrics.
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

        # build the task similarity graph: the similarity between two tasks
        # is 1 / (1 + distance), so the most similar neighbors are the
        # nearest neighbors in task descriptor space
        num_tasks = task_descriptors.shape[0]
        num_neighbors = min(self._num_neighbors, num_tasks - 1)
        distances = jnp.sum(
            jnp.square(task_descriptors[:, None, :] - task_descriptors[None, :, :]),
            axis=-1,
        )
        # exclude self-loops
        distances = distances + jnp.diag(jnp.full(num_tasks, jnp.inf))
        _, neighbor_indices = jax.lax.top_k(-distances, num_neighbors)
        monet_state = MONETState(neighbor_indices=neighbor_indices)

        # calculate the initial metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, monet_state, metrics

    def _select_neighbors(
        self,
        repertoire: MONETRepertoire,
        neighbor_candidates: jax.Array,
        key: RNGKey,
    ) -> jax.Array:
        """Select, for each focal task, the neighbor used for social
        learning.

        Args:
            repertoire: the MONET repertoire
            neighbor_candidates: neighbors of each focal task, sorted by
                decreasing similarity, of shape (batch_size, num_neighbors)
            key: a jax PRNG random key

        Returns:
            the indices of the selected neighbor tasks, of shape
            (batch_size,)
        """
        batch_size, num_neighbors = neighbor_candidates.shape
        neighbor_fitnesses = repertoire.fitnesses[neighbor_candidates, 0]

        if self._neighbor_strategy == "best_fitness":
            columns = jnp.argmax(neighbor_fitnesses, axis=1)
        elif self._neighbor_strategy == "random":
            columns = jax.random.randint(key, (batch_size,), 0, num_neighbors)
        elif self._neighbor_strategy == "most_similar":
            top_k = min(self._top_k_similar, num_neighbors)
            columns = jax.random.randint(key, (batch_size,), 0, top_k)
        elif self._neighbor_strategy == "least_similar":
            top_k = min(self._top_k_similar, num_neighbors)
            columns = (
                num_neighbors - 1 - jax.random.randint(key, (batch_size,), 0, top_k)
            )
        else:  # fitness_proportional
            occupied = neighbor_fitnesses > -jnp.inf
            min_fitness = jnp.min(
                jnp.where(occupied, neighbor_fitnesses, jnp.inf),
                axis=1,
                keepdims=True,
            )
            weights = jnp.where(occupied, neighbor_fitnesses - min_fitness + 1e-6, 0.0)
            # fall back to a uniform choice when no neighbor has an elite
            logits = jnp.where(
                weights > 0.0, jnp.log(jnp.maximum(weights, 1e-30)), -jnp.inf
            )
            logits = jnp.where(
                jnp.sum(weights, axis=1, keepdims=True) > 0.0,
                logits,
                jnp.zeros_like(logits),
            )
            columns = jax.random.categorical(key, logits, axis=1)

        return neighbor_candidates[jnp.arange(batch_size), columns]

    def _apply_operators(
        self,
        operators: Tuple[str, ...],
        offsprings: Tuple[Genotype, ...],
        key: RNGKey,
        batch_size: int,
    ) -> Genotype:
        """Select, for each offspring, the result of one of the operators,
        uniformly at random."""
        if len(operators) == 1:
            return offsprings[0]
        operator_indices = jax.random.randint(key, (batch_size,), 0, len(operators))
        result = offsprings[0]
        for i in range(1, len(operators)):
            result = _tree_where(operator_indices == i, offsprings[i], result)
        return result

    def _individual_learning(
        self, x: Genotype, key: RNGKey, batch_size: int
    ) -> Genotype:
        """Apply individual learning (mutation) to the focal elites."""
        offsprings = []
        for operator in self._individual_learning_ops:
            key, subkey = jax.random.split(key)
            if operator == "gaussian_mutation":
                offsprings.append(
                    _gaussian_mutation(
                        x, subkey, self._mutation_std, self._minval, self._maxval
                    )
                )
            else:  # polynomial_mutation
                offsprings.append(
                    _polynomial_mutation(
                        x, subkey, self._polynomial_eta, self._minval, self._maxval
                    )
                )
        key, subkey = jax.random.split(key)
        return self._apply_operators(
            self._individual_learning_ops, tuple(offsprings), subkey, batch_size
        )

    def _social_learning(
        self, x: Genotype, y: Genotype, key: RNGKey, batch_size: int
    ) -> Genotype:
        """Apply social learning (crossover with a neighbor elite) to the
        focal elites."""
        offsprings = []
        for operator in self._social_learning_ops:
            key, subkey = jax.random.split(key)
            if operator == "sbx":
                offsprings.append(
                    _sbx_crossover(
                        x, y, subkey, self._sbx_eta, self._minval, self._maxval
                    )
                )
            else:  # iso_dd
                offsprings.append(
                    _iso_dd_crossover(
                        x,
                        y,
                        subkey,
                        self._iso_sigma,
                        self._line_sigma,
                        self._minval,
                        self._maxval,
                    )
                )
        key, subkey = jax.random.split(key)
        return self._apply_operators(
            self._social_learning_ops, tuple(offsprings), subkey, batch_size
        )

    def ask(
        self,
        repertoire: MONETRepertoire,
        monet_state: MONETState,
        key: RNGKey,
    ) -> Tuple[Genotype, jax.Array, ExtraScores]:
        """
        Generate a new batch of genotypes: sample the focal tasks, apply
        individual or social learning to their elites and return the
        offsprings together with the indices of the focal tasks on which
        they must be evaluated.

        Args:
            repertoire: the MONET repertoire
            monet_state: the MONET state, holding the task similarity graph
            key: a jax PRNG random key

        Returns:
            a new batch of genotypes, the indices of the tasks on which they
            must be evaluated and an empty extra information dictionary
        """
        batch_size = self._batch_size
        num_tasks = repertoire.centroids.shape[0]

        # sample the focal tasks and gather their elites
        key, subkey = jax.random.split(key)
        task_indices = jax.random.randint(subkey, (batch_size,), 0, num_tasks)
        x = jax.tree.map(lambda g: g[task_indices], repertoire.genotypes)
        focal_occupied = repertoire.fitnesses[task_indices, 0] > -jnp.inf

        # individual learning vs social learning
        key, subkey = jax.random.split(key)
        individual = jax.random.bernoulli(subkey, self._p_ind, (batch_size,))

        # select the neighbors and gather their elites
        key, subkey = jax.random.split(key)
        neighbor_indices = self._select_neighbors(
            repertoire, monet_state.neighbor_indices[task_indices], subkey
        )
        y = jax.tree.map(lambda g: g[neighbor_indices], repertoire.genotypes)
        neighbor_occupied = repertoire.fitnesses[neighbor_indices, 0] > -jnp.inf

        # individual learning: mutation of the focal elite
        key, subkey = jax.random.split(key)
        individual_offspring = self._individual_learning(x, subkey, batch_size)

        # social learning: crossover with the neighbor elite, with a
        # fallback to a Gaussian mutation when the neighbor has no elite
        key, subkey = jax.random.split(key)
        social_offspring = self._social_learning(x, y, subkey, batch_size)
        key, subkey = jax.random.split(key)
        social_offspring = _tree_where(
            neighbor_occupied,
            social_offspring,
            _gaussian_mutation(
                x, subkey, self._mutation_std, self._minval, self._maxval
            ),
        )

        genotypes = _tree_where(individual, individual_offspring, social_offspring)

        # focal tasks without elite: copy the neighbor elite (when social
        # learning is applied and the neighbor has one) or sample a random
        # solution
        key, subkey = jax.random.split(key)
        random_genotypes = _tree_uniform(x, subkey, self._minval, self._maxval)
        if self._init_strategy == "copy":
            init_genotypes = _tree_where(
                (~individual) & neighbor_occupied, y, random_genotypes
            )
        else:
            init_genotypes = random_genotypes
        genotypes = _tree_where(focal_occupied, genotypes, init_genotypes)

        return genotypes, task_indices, {}

    def tell(
        self,
        genotypes: Genotype,
        fitnesses: Fitness,
        task_indices: jax.Array,
        repertoire: MONETRepertoire,
        monet_state: MONETState,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Tuple[MONETRepertoire, MONETState, Metrics]:
        """
        Add new genotypes to the repertoire.

        Args:
            genotypes: new genotypes to add to the repertoire
            fitnesses: fitnesses of the new genotypes on their tasks
            task_indices: indices of the tasks on which the new genotypes
                have been evaluated
            repertoire: the MONET repertoire
            monet_state: the MONET state
            extra_scores: extra scores of the new genotypes
        """
        if extra_scores is None:
            extra_scores = {}

        fitnesses = jnp.ravel(fitnesses)
        task_descriptors = repertoire.centroids[task_indices]

        # add genotypes to the repertoire: as the descriptor of an offspring
        # is exactly the descriptor of its task, it lands in the task's cell
        repertoire = repertoire.add(
            genotypes, task_descriptors, fitnesses, extra_scores
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, monet_state, metrics

    def update(
        self,
        repertoire: MONETRepertoire,
        monet_state: MONETState,
        key: RNGKey,
    ) -> Tuple[MONETRepertoire, MONETState, Metrics]:
        """
        Performs one iteration of the MONET algorithm.
        1. A batch of focal tasks is sampled uniformly at random.
        2. The elite of each focal task is either mutated (individual
            learning) or crossed-over with the elite of one of the
            neighbors of the task in the similarity graph (social
            learning).
        3. The offsprings are scored on their focal tasks and added to the
            repertoire if their fitness is greater than or equal to the
            fitness of the current elite.

        Args:
            repertoire: the MONET repertoire
            monet_state: the MONET state, holding the task similarity graph
            key: a jax PRNG random key

        Returns:
            the updated MONET repertoire, the MONET state and metrics about
            the repertoire
        """
        if self._scoring_function is None:
            raise ValueError("Scoring function is not set.")

        # generate offsprings on their focal tasks
        key, subkey = jax.random.split(key)
        genotypes, task_indices, _ = self.ask(repertoire, monet_state, subkey)

        # score the offsprings on the focal tasks
        key, subkey = jax.random.split(key)
        fitnesses, extra_scores = self._scoring_function(
            genotypes, repertoire.centroids[task_indices], subkey
        )

        return self.tell(
            genotypes=genotypes,
            fitnesses=fitnesses,
            task_indices=task_indices,
            repertoire=repertoire,
            monet_state=monet_state,
            extra_scores=extra_scores,
        )

    def scan_update(
        self,
        carry: Tuple[MONETRepertoire, MONETState, RNGKey],
        _: Any,
    ) -> Tuple[Tuple[MONETRepertoire, MONETState, RNGKey], Metrics]:
        """Rewrites the update function in a way that makes it compatible
        with the jax.lax.scan primitive.

        Args:
            carry: a tuple containing the repertoire, the MONET state and a
                random key.
            _: unused element, necessary to respect jax.lax.scan API.

        Returns:
            The updated repertoire and MONET state, with a new random key
            and metrics.
        """
        repertoire, monet_state, key = carry
        key, subkey = jax.random.split(key)
        (
            repertoire,
            monet_state,
            metrics,
        ) = self.update(
            repertoire,
            monet_state,
            subkey,
        )

        return (repertoire, monet_state, key), metrics
