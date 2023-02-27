from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import flax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Observation,
    RNGKey,
)


@partial(jax.jit, static_argnames=("k_nn",))
def get_cells_indices(
    batch_of_descriptors: jnp.ndarray, centroids: jnp.ndarray, k_nn: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns the array of cells indices for a batch of descriptors
    given the centroids of the grid.

    Args:
        batch_of_descriptors: a batch of descriptors
            of shape (batch_size, num_descriptors)
        centroids: centroids array of shape (num_centroids, num_descriptors)

    Returns:
        the indices of the centroids corresponding to each vector of descriptors
            in the batch with shape (batch_size,)
    """

    def _get_cells_indices(
        descriptors: jnp.ndarray,
        centroids: jnp.ndarray,
        k_nn: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        set_of_descriptors of shape (1, num_descriptors)
        centroids of shape (num_centroids, num_descriptors)
        """

        # distances = jnp.sum(jnp.square(jnp.subtract(descriptors, centroids)), axis=-1)
        distances = jax.vmap(jnp.linalg.norm)(descriptors - centroids)
        ## Negating distances because we want the smallest ones
        min_dist, min_args = jax.lax.top_k(-1 * distances, k_nn)
        # return jnp.argmin(distances),jnp.min(distances)
        return min_args, -1 * min_dist

    # func = jax.vmap(lambda x: _get_cells_indices(x, centroids,k_nn),in_axes=(0,None,None,))
    func = jax.vmap(
        _get_cells_indices,
        in_axes=(
            0,
            None,
            None,
        ),
    )

    # return func(batch_of_descriptors)
    return func(batch_of_descriptors, centroids, k_nn)


@jax.jit
def intra_batch_comp(
    normed,
    current_index,
    normed_all,
    eval_scores,
    l_value,
):

    ## Check for individuals that are Nans, we remove them at the end
    not_existent = jnp.where((jnp.isnan(normed)).any(), True, False)
    ## Fill in Nans to do computations
    normed = jnp.where(jnp.isnan(normed), jnp.full(normed.shape[-1], jnp.inf), normed)
    eval_scores = jnp.where(
        jnp.isinf(eval_scores), jnp.full(eval_scores.shape[-1], jnp.nan), eval_scores
    )
    ## If we do not use a fitness (i.e same fitness everywhere, we create a virtual fitness function to add individuals with the same bd)
    additional_score = jnp.where(
        jnp.nanmax(eval_scores) == jnp.nanmin(eval_scores), 1.0, 0.0
    )
    additional_scores = jnp.linspace(0.0, additional_score, num=eval_scores.shape[0])
    ## Add scores to empty individuals
    eval_scores = jnp.where(
        jnp.isnan(eval_scores), jnp.full(eval_scores.shape[0], -jnp.inf), eval_scores
    )
    ##Virtual eval_scores
    eval_scores = eval_scores + additional_scores
    ## For each point we check what other points are the closest ones.
    knn_relevant_scores, knn_relevant_indices = jax.lax.top_k(
        -1 * jax.vmap(jnp.linalg.norm)(normed - normed_all), eval_scores.shape[0]
    )
    ## We negated the scores to use top_k so we reverse it.
    knn_relevant_scores = knn_relevant_scores * -1

    ##Check if the individual is close enough to compare (under l-value)
    fitness = jnp.where(jnp.squeeze(knn_relevant_scores < l_value), True, False)
    ## We want to eliminate the same individual (distance 0)
    # fitness = jnp.where(knn_relevant_scores==0.0,False,fitness)
    fitness = jnp.where(knn_relevant_indices == current_index, False, fitness)
    current_fitness = jnp.squeeze(
        eval_scores.at[knn_relevant_indices.at[0].get()].get()
    )

    ## Is the fitness of the other individual higher?
    ## If both are True then we discard the current individual since this individual would be replaced by the better one.
    discard_indiv = jnp.logical_and(
        jnp.where(
            eval_scores.at[knn_relevant_indices].get() > current_fitness, True, False
        ),
        fitness,
    ).any()
    ## Discard Individuals with Nans as their BD (mainly for the readdition where we have NaN bds)
    discard_indiv = jnp.logical_or(discard_indiv, not_existent)

    ## Negate to know if we keep the individual
    return jnp.logical_not(discard_indiv)


@jax.jit
def intra_batch_comp_relevant(
    normed,
    current_index,
    normed_all,
    eval_scores,
    relevant_l_values,
):

    ## Check for individuals that are Nans, we remove them at the end
    not_existent = jnp.where((jnp.isnan(normed)).any(), True, False)
    ## Fill in Nans to do computations
    normed = jnp.where(jnp.isnan(normed), jnp.full(normed.shape[-1], jnp.inf), normed)
    eval_scores = jnp.where(
        jnp.isinf(eval_scores), jnp.full(eval_scores.shape[-1], jnp.nan), eval_scores
    )
    ## If we do not use a fitness (i.e same fitness everywhere, we create a virtual fitness function to add individuals with the same bd)
    additional_score = jnp.where(
        jnp.nanmax(eval_scores) == jnp.nanmin(eval_scores), 1.0, 0.0
    )
    additional_scores = jnp.linspace(0.0, additional_score, num=eval_scores.shape[0])
    ## Add scores to empty individuals
    eval_scores = jnp.where(
        jnp.isnan(eval_scores), jnp.full(eval_scores.shape[0], -jnp.inf), eval_scores
    )
    ##Virtual eval_scores
    eval_scores = eval_scores + additional_scores
    ## For each point we check what other points are the closest ones.
    knn_relevant_scores, knn_relevant_indices = jax.lax.top_k(
        -1 * jax.vmap(jnp.linalg.norm)(normed - normed_all), eval_scores.shape[0]
    )
    ## We negated the scores to use top_k so we reverse it.
    knn_relevant_scores = knn_relevant_scores * -1

    ##Check if the individual is close enough to compare (under l-value)
    fitness = jnp.where(
        jnp.squeeze(knn_relevant_scores < relevant_l_values), True, False
    )
    ## We want to eliminate the same individual (distance 0)
    # fitness = jnp.where(knn_relevant_scores==0.0,False,fitness)
    fitness = jnp.where(knn_relevant_indices == current_index, False, fitness)
    current_fitness = jnp.squeeze(
        eval_scores.at[knn_relevant_indices.at[0].get()].get()
    )

    ## Is the fitness of the other individual higher?
    ## If both are True then we discard the current individual since this individual would be replaced by the better one.
    discard_indiv = jnp.logical_and(
        jnp.where(
            eval_scores.at[knn_relevant_indices].get() > current_fitness, True, False
        ),
        fitness,
    ).any()
    ## Discard Individuals with Nans as their BD (mainly for the readdition where we have NaN bds)
    discard_indiv = jnp.logical_or(discard_indiv, not_existent)

    ## Negate to know if we keep the individual
    return jnp.logical_not(discard_indiv)


class UnstructuredRepertoire(flax.struct.PyTreeNode):
    """
    Class for the unstructured repertoire in Map Elites.

    Args:
        genotypes: a PyTree containing all the genotypes in the repertoire ordered
            by the centroids. Each leaf has a shape (num_centroids, num_features). The
            PyTree can be a simple Jax array or a more complex nested structure such
            as to represent parameters of neural network in Flax.
        fitnesses: an array that contains the fitness of solutions in each cell of the
            repertoire, ordered by centroids. The array shape is (num_centroids,).
        descriptors: an array that contains the descriptors of solutions in each cell
            of the repertoire, ordered by centroids. The array shape
            is (num_centroids, num_descriptors).
        centroids: an array the contains the centroids of the tesselation. The array
            shape is (num_centroids, num_descriptors).
    """

    genotypes: Genotype
    fitnesses: Fitness
    descriptors: Descriptor
    centroids: Centroid
    observations: ExtraScores
    ages: jnp.ndarray
    l_value: jnp.ndarray

    def save(self, path: str = "./") -> None:
        """Saves the grid on disk in the form of .npy files.

        Flattens the genotypes to store it with .npy format. Supposes that
        a user will have access to the reconstruction function when loading
        the genotypes.

        Args:
            path: Path where the data will be saved. Defaults to "./".
        """

        def flatten_genotype(genotype: Genotype) -> jnp.ndarray:
            flatten_genotype, _unravel_pytree = ravel_pytree(genotype)
            return flatten_genotype

        # flatten all the genotypes
        flat_genotypes = jax.vmap(flatten_genotype)(self.genotypes)

        # save data
        jnp.save(path + "genotypes.npy", flat_genotypes)
        jnp.save(path + "fitnesses.npy", self.fitnesses)
        jnp.save(path + "descriptors.npy", self.descriptors)
        jnp.save(path + "centroids.npy", self.centroids)
        jnp.save(path + "observations.npy", self.observations)
        jnp.save(path + "l_value.npy", self.l_value)
        jnp.save(path + "ages.npy", self.ages)

    @classmethod
    def load(
        cls, reconstruction_fn: Callable, path: str = "./"
    ) -> UnstructuredRepertoire:
        """Loads a MAP Elites Grid.

        Args:
            reconstruction_fn: Function to reconstruct a PyTree
                from a flat array.
            path: Path where the data is saved. Defaults to "./".

        Returns:
            A MAP Elites Repertoire.
        """

        flat_genotypes = jnp.load(path + "genotypes.npy")
        genotypes = jax.vmap(reconstruction_fn)(flat_genotypes)

        fitnesses = jnp.load(path + "fitnesses.npy")
        descriptors = jnp.load(path + "descriptors.npy")
        centroids = jnp.load(path + "centroids.npy")
        observations = jnp.load(path + "observations.npy")
        l_value = jnp.load(path + "l_value.npy")
        ages = jnp.load(path + "ages.npy")

        return UnstructuredRepertoire(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            observations=observations,
            l_value=l_value,
            ages=ages,
        )

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_observations: Observation,
    ) -> UnstructuredRepertoire:

        ## We need to replace all the descriptors that are not filled with jnp inf
        filtered_descriptors = jnp.where(
            jnp.expand_dims((self.fitnesses == -jnp.inf), axis=-1),
            jnp.full(self.descriptors.shape[-1], fill_value=jnp.inf),
            self.descriptors,
        )

        batch_of_indices, batch_of_distances = get_cells_indices(
            batch_of_descriptors, filtered_descriptors, 2
        )

        second_neighbours = batch_of_distances.at[
            ..., 1
        ].get()  # Save the second nearest neighbours to check a condition
        batch_of_indices = batch_of_indices.at[
            ..., 0
        ].get()  ## Keep the Nearest neighbours
        batch_of_distances = batch_of_distances.at[
            ..., 0
        ].get()  ## Keep the Nearest neighbours

        # We remove individuals that are too close to the second nn.
        # This avoids having clusters of individuals after adding them.
        not_novel_enough = jnp.where(
            jnp.squeeze(second_neighbours <= self.l_value), True, False
        )

        # batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)
        batch_of_fitnesses = jnp.expand_dims(batch_of_fitnesses, axis=-1)
        batch_of_observations = jnp.expand_dims(batch_of_observations, axis=-1)

        num_centroids = self.centroids.shape[0]

        ### TODO Doesn't Work if Archive is full. Need to use the closest individuals in that case.
        empty_indexes = jnp.squeeze(
            jnp.nonzero(
                jnp.where(jnp.isinf(self.fitnesses), 1, 0),
                size=batch_of_indices.shape[0],
                fill_value=-1,
            )[0]
        )
        batch_of_indices = jnp.where(
            jnp.squeeze(batch_of_distances <= self.l_value),
            jnp.squeeze(batch_of_indices),
            -1,
        )

        sorted_bds = jax.lax.top_k(
            -1 * batch_of_indices.squeeze(), batch_of_indices.shape[0]
        )[
            1
        ]  ## We get all the indices of the empty bds first and then the filled ones (because of -1)
        batch_of_indices = jnp.where(
            jnp.squeeze(batch_of_distances.at[sorted_bds].get() <= self.l_value),
            batch_of_indices.at[sorted_bds].get(),
            empty_indexes,
        )

        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)

        ## ReIndexing of all the inputs to the correct sorted way
        batch_of_distances = batch_of_distances.at[sorted_bds].get()
        batch_of_descriptors = batch_of_descriptors.at[sorted_bds].get()
        batch_of_genotypes = jax.tree_map(
            lambda x: x.at[sorted_bds].get(), batch_of_genotypes
        )
        # obs = obs.at[sorted_bds].get()
        batch_of_fitnesses = batch_of_fitnesses.at[sorted_bds].get()
        batch_of_observations = batch_of_observations.at[sorted_bds].get()
        not_novel_enough = not_novel_enough.at[sorted_bds].get()
        # dead = dead.at[sorted_bds].get()

        ## Check to find Individuals with same BD within the Batch
        keep_indiv = jax.jit(
            jax.vmap(intra_batch_comp, in_axes=(0, 0, None, None, None), out_axes=(0))
        )(
            batch_of_descriptors.squeeze(),
            jnp.arange(
                0, batch_of_descriptors.shape[0], 1
            ),  ## We do this to keep track of where we are in the batch to assure right comparisons
            batch_of_descriptors.squeeze(),
            batch_of_fitnesses.squeeze(),
            self.l_value,
        )

        keep_indiv = jnp.logical_and(keep_indiv, jnp.logical_not(not_novel_enough))

        # keep_indiv = jax.vmap(intra_batch_comp, in_axes=(0,0,None,None,None), out_axes=(0))(batch_of_descriptors.squeeze(),jnp.arange(0,batch_of_descriptors.shape[0],1),batch_of_descriptors.squeeze(),batch_of_fitnesses.squeeze(),self.l_value)
        # get fitness segment max
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices.astype(jnp.int32).squeeze(),
            num_segments=num_centroids,
        )

        cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0)

        # put dominated fitness to -jnp.inf
        batch_of_fitnesses = jnp.where(
            batch_of_fitnesses == cond_values, x=batch_of_fitnesses, y=-jnp.inf
        )

        # get addition condition
        grid_fitnesses = jnp.expand_dims(self.fitnesses, axis=-1)
        current_fitnesses = jnp.take_along_axis(grid_fitnesses, batch_of_indices, 0)
        addition_condition = batch_of_fitnesses > current_fitnesses
        addition_condition = jnp.logical_and(
            addition_condition, jnp.expand_dims(keep_indiv, axis=-1)
        )

        # assign fake position when relevant : num_centroids is out of bounds
        batch_of_indices = jnp.where(
            addition_condition, x=batch_of_indices, y=num_centroids
        )

        # create new grid
        new_grid_genotypes = jax.tree_map(
            lambda grid_genotypes, new_genotypes: grid_genotypes.at[
                batch_of_indices.squeeze()
            ].set(new_genotypes),
            self.genotypes,
            batch_of_genotypes,
        )

        # compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[batch_of_indices.squeeze()].set(
            batch_of_fitnesses.squeeze()
        )
        new_descriptors = self.descriptors.at[batch_of_indices.squeeze()].set(
            batch_of_descriptors.squeeze()
        )

        new_observations = self.observations.at[batch_of_indices.squeeze()].set(
            batch_of_observations.squeeze()
        )

        new_ages = self.ages.at[batch_of_indices.squeeze()].set(0.0) + 1

        return UnstructuredRepertoire(
            genotypes=new_grid_genotypes,
            fitnesses=new_fitnesses.squeeze(),
            descriptors=new_descriptors.squeeze(),
            centroids=new_descriptors.squeeze(),
            observations=new_observations.squeeze(),
            l_value=self.l_value,
            ages=new_ages,
        )

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:
        """
        Sample elements in the grid.

        Args:
            random_key: a jax PRNG random key
            num_samples: the number of elements to be sampled

        Returns:
            samples: a batch of genotypes sampled in the repertoire
            random_key: an updated jax PRNG random key
        """

        random_key, sub_key = jax.random.split(random_key)
        grid_empty = self.fitnesses == -jnp.inf
        p = (1.0 - grid_empty) / jnp.sum(1.0 - grid_empty)

        samples = jax.tree_map(
            lambda x: jax.random.choice(sub_key, x, shape=(num_samples,), p=p),
            self.genotypes,
        )

        return samples, random_key

    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        observations: ExtraScores,
        l_value: jnp.ndarray,
        ages: Optional[jnp.ndarray] = None,
    ) -> UnstructuredRepertoire:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Note: this function has been kept outside of the object MapElites, so it can
        be called easily called from other modules.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            fitnesses: fitness of the initial genotypes of shape (batch_size,)
            descriptors: descriptors of the initial genotypes
                of shape (batch_size, num_descriptors)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)

        Returns:
            an initialized MAP-Elite repertoire
        """

        # Initialize grid with default values
        num_centroids = centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)
        default_genotypes = jax.tree_map(
            lambda x: jnp.full(
                shape=(num_centroids,) + x.shape[1:], fill_value=jnp.nan
            ),
            genotypes,
        )
        default_descriptors = jnp.zeros(shape=(num_centroids, centroids.shape[-1]))

        default_observations = jnp.full(
            shape=(num_centroids,) + observations.shape[1:], fill_value=jnp.nan
        )

        if ages is None:
            ages = jnp.zeros(shape=num_centroids)

        repertoire = UnstructuredRepertoire(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
            observations=default_observations,
            l_value=l_value,
            ages=ages,
        )

        # return new_repertoire  # type: ignore
        return repertoire.add(genotypes, descriptors, fitnesses, observations)

    @jax.jit
    def add_relevant(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_observations: Observation,
        proximity_scores: jnp.ndarray,
    ) -> UnstructuredRepertoire:

        # Calculating new l values
        new_l_values = self.l_value / proximity_scores

        # We need to replace all the descriptors that are not filled with jnp inf
        filtered_descriptors = jnp.where(
            jnp.expand_dims((self.fitnesses == -jnp.inf), axis=-1),
            jnp.full(self.descriptors.shape[-1], fill_value=jnp.inf),
            self.descriptors,
        )

        batch_of_indices, batch_of_distances = get_cells_indices(
            batch_of_descriptors, filtered_descriptors, 2
        )

        second_neighbours = batch_of_distances.at[
            ..., 1
        ].get()  # Save the second nearest neighbours to check a condition
        batch_of_indices = batch_of_indices.at[
            ..., 0
        ].get()  ## Keep the Nearest neighbours
        batch_of_distances = batch_of_distances.at[
            ..., 0
        ].get()  ## Keep the Nearest neighbours

        # We remove individuals that are too close to the second nn.
        # This avoids having clusters of individuals after adding them.
        not_novel_enough = jnp.where(
            jnp.squeeze(second_neighbours <= new_l_values), True, False
        )

        # batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)
        batch_of_fitnesses = jnp.expand_dims(batch_of_fitnesses, axis=-1)
        batch_of_observations = jnp.expand_dims(batch_of_observations, axis=-1)

        num_centroids = self.centroids.shape[0]

        ### TODO Doesn't Work if Archive is full. Need to use the closest individuals in that case.
        empty_indexes = jnp.squeeze(
            jnp.nonzero(
                jnp.where(jnp.isinf(self.fitnesses), 1, 0),
                size=batch_of_indices.shape[0],
                fill_value=-1,
            )[0]
        )
        batch_of_indices = jnp.where(
            jnp.squeeze(batch_of_distances <= new_l_values),
            jnp.squeeze(batch_of_indices),
            -1,
        )

        sorted_bds = jax.lax.top_k(
            -1 * batch_of_indices.squeeze(), batch_of_indices.shape[0]
        )[
            1
        ]  ## We get all the indices of the empty bds first and then the filled ones (because of -1)
        batch_of_indices = jnp.where(
            jnp.squeeze(
                batch_of_distances.at[sorted_bds].get()
                <= new_l_values.at[sorted_bds].get()
            ),
            batch_of_indices.at[sorted_bds].get(),
            empty_indexes,
        )

        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)

        ## ReIndexing of all the inputs to the correct sorted way
        batch_of_distances = batch_of_distances.at[sorted_bds].get()
        batch_of_descriptors = batch_of_descriptors.at[sorted_bds].get()
        batch_of_genotypes = jax.tree_map(
            lambda x: x.at[sorted_bds].get(), batch_of_genotypes
        )
        # obs = obs.at[sorted_bds].get()
        batch_of_fitnesses = batch_of_fitnesses.at[sorted_bds].get()
        batch_of_observations = batch_of_observations.at[sorted_bds].get()
        not_novel_enough = not_novel_enough.at[sorted_bds].get()
        new_l_values = new_l_values.at[sorted_bds].get()
        # dead = dead.at[sorted_bds].get()

        # filtered_l = jnp.where(new_l_values>self.l_value,self.l_value,new_l_values)
        ## Check to find Individuals with same BD within the Batch
        keep_indiv = jit(
            jax.vmap(intra_batch_comp, in_axes=(0, 0, None, None, 0), out_axes=(0))
        )(
            batch_of_descriptors.squeeze(),
            jnp.arange(
                0, batch_of_descriptors.shape[0], 1
            ),  ## We do this to keep track of where we are in the batch to assure right comparisons
            batch_of_descriptors.squeeze(),
            batch_of_fitnesses.squeeze(),
            new_l_values,
        )

        # keep_indiv = jnp.logical_and(keep_indiv,jnp.logical_not(not_novel_enough))

        # keep_indiv = jax.vmap(intra_batch_comp, in_axes=(0,0,None,None,None), out_axes=(0))(batch_of_descriptors.squeeze(),jnp.arange(0,batch_of_descriptors.shape[0],1),batch_of_descriptors.squeeze(),batch_of_fitnesses.squeeze(),self.l_value)
        # get fitness segment max
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices.astype(jnp.int32).squeeze(),
            num_segments=num_centroids,
        )

        cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0)

        # put dominated fitness to -jnp.inf
        batch_of_fitnesses = jnp.where(
            batch_of_fitnesses == cond_values, x=batch_of_fitnesses, y=-jnp.inf
        )

        # get addition condition
        grid_fitnesses = jnp.expand_dims(self.fitnesses, axis=-1)
        current_fitnesses = jnp.take_along_axis(grid_fitnesses, batch_of_indices, 0)
        addition_condition = batch_of_fitnesses > current_fitnesses
        addition_condition = jnp.logical_and(
            addition_condition, jnp.expand_dims(keep_indiv, axis=-1)
        )
        print(addition_condition)
        print(batch_of_indices)
        print(batch_of_descriptors)
        print(batch_of_distances)
        print(new_l_values)

        # assign fake position when relevant : num_centroids is out of bounds
        batch_of_indices = jnp.where(
            addition_condition, x=batch_of_indices, y=num_centroids
        )

        # create new grid
        new_grid_genotypes = jax.tree_map(
            lambda grid_genotypes, new_genotypes: grid_genotypes.at[
                batch_of_indices.squeeze()
            ].set(new_genotypes),
            self.genotypes,
            batch_of_genotypes,
        )

        # compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[batch_of_indices.squeeze()].set(
            batch_of_fitnesses.squeeze()
        )
        new_descriptors = self.descriptors.at[batch_of_indices.squeeze()].set(
            batch_of_descriptors.squeeze()
        )

        new_observations = self.observations.at[batch_of_indices.squeeze()].set(
            batch_of_observations.squeeze()
        )

        new_ages = self.ages.at[batch_of_indices.squeeze()].set(0.0) + 1

        return UnstructuredRepertoire(
            genotypes=new_grid_genotypes,
            fitnesses=new_fitnesses.squeeze(),
            descriptors=new_descriptors.squeeze(),
            centroids=new_descriptors.squeeze(),
            observations=new_observations.squeeze(),
            l_value=self.l_value,
            ages=new_ages,
        )

    @classmethod
    def init_relevant(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        observations: ExtraScores,
        l_value: float,
        proximity_scores: jnp.ndarray,
        ages: Optional[jnp.ndarray] = None,
    ) -> UnstructuredRepertoire:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Note: this function has been kept outside of the object MapElites, so it can
        be called easily called from other modules.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            fitnesses: fitness of the initial genotypes of shape (batch_size,)
            descriptors: descriptors of the initial genotypes
                of shape (batch_size, num_descriptors)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)

        Returns:
            an initialized MAP-Elite repertoire
        """

        # Initialize grid with default values
        num_centroids = centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)
        default_genotypes = jax.tree_map(
            lambda x: jnp.full(
                shape=(num_centroids,) + x.shape[1:], fill_value=jnp.nan
            ),
            genotypes,
        )
        default_descriptors = jnp.zeros(shape=(num_centroids, centroids.shape[-1]))

        default_observations = jnp.full(
            shape=(num_centroids,) + observations.shape[1:], fill_value=jnp.nan
        )

        if ages is None:
            ages = jnp.zeros(shape=num_centroids)
        repertoire = UnstructuredRepertoire(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
            observations=default_observations,
            l_value=l_value,
            ages=ages,
        )

        # return new_repertoire
        return repertoire.add_relevant(
            genotypes, descriptors, fitnesses, observations, proximity_scores
        )
