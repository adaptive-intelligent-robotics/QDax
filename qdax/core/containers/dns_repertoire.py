"""Dominated Novelty Search (DNS) repertoire.

This container stores a flat population (no tessellation) and uses dominated
novelty as a meta-fitness to select survivors when adding new individuals.

Dominated novelty is defined as the average distance in descriptor space to the
K-nearest neighbors that are fitter (have greater or equal fitness).
"""

from __future__ import annotations

from typing import Optional, Tuple

import flax.struct
import jax
import jax.numpy as jnp

from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype


def _novelty_and_dominated_novelty(
    fitness: jax.Array,
    descriptor: jax.Array,
    novelty_k: int,
    dominated_novelty_k: int,
) -> Tuple[jax.Array, jax.Array]:
    """Compute novelty and dominated novelty.

    Args:
        fitness: shape (N,) fitness values, higher is better.
        descriptor: shape (N, D) descriptors.
        novelty_k: number of neighbors for novelty.
        dominated_novelty_k: number of fitter-neighbors for dominated novelty.

    Returns:
        novelty: shape (N,)
        dominated_novelty: shape (N,)
    """

    valid = fitness != -jnp.inf

    # Neighbor mask excluding self
    neighbor = valid[:, None] & valid[None, :]
    neighbor = jnp.fill_diagonal(neighbor, False, inplace=False)

    # Fitter-or-equal mask
    fitter = fitness[:, None] <= fitness[None, :]
    fitter = jnp.where(neighbor, fitter, False)

    # Pairwise distances
    distance = jnp.linalg.norm(descriptor[:, None, :] - descriptor[None, :, :], axis=-1)
    distance = jnp.where(neighbor, distance, jnp.inf)

    # Distances to fitter neighbors only
    distance_fitter = jnp.where(fitter, distance, jnp.inf)

    # Novelty: mean distance to k nearest neighbors
    values, indices = jax.vmap(lambda x: jax.lax.top_k(-x, novelty_k))(distance)
    novelty = jnp.mean(
        -values,
        axis=-1,
        where=jnp.take_along_axis(neighbor, indices, axis=-1),
    )

    # Dominated novelty: mean distance to k nearest fitter neighbors
    values_fit, indices_fit = jax.vmap(
        lambda x: jax.lax.top_k(-x, dominated_novelty_k)
    )(distance_fitter)
    dominated_novelty = jnp.mean(
        -values_fit,
        axis=-1,
        where=jnp.take_along_axis(fitter, indices_fit, axis=-1),
    )

    return novelty, dominated_novelty


class DominatedNoveltyRepertoire(GARepertoire):
    """Repertoire that keeps the individuals with highest dominated novelty.

    Args:
        genotypes: population genotypes with shape (population_size, ...)
        fitnesses: population fitnesses with shape (population_size, fitness_dim)
        descriptors: population descriptors with shape (population_size, D)
        k: number of neighbors for novelty and dominated novelty
        extra_scores: extra scores resulting from the evaluation of the genotypes
        keys_extra_scores: keys of the extra scores to store in the repertoire
    """

    descriptors: Descriptor
    k: int = flax.struct.field(pytree_node=False)

    @jax.jit
    def add(  # type: ignore
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> DominatedNoveltyRepertoire:
        """Add a batch and keep the top individuals by dominated novelty.

        Parents and offsprings are gathered and only the population_size
        best according to dominated novelty are kept.
        """

        if batch_of_extra_scores is None:
            batch_of_extra_scores = {}

        filtered_batch_of_extra_scores = self.filter_extra_scores(batch_of_extra_scores)

        batch_of_fitnesses = jnp.reshape(
            batch_of_fitnesses, (batch_of_fitnesses.shape[0], 1)
        )

        # Gather candidates
        candidates_genotypes = jax.tree.map(
            lambda x, y: jnp.concatenate((x, y), axis=0),
            self.genotypes,
            batch_of_genotypes,
        )
        candidates_fitnesses = jnp.concatenate(
            (self.fitnesses, batch_of_fitnesses), axis=0
        )
        candidates_descriptors = jnp.concatenate(
            (self.descriptors, batch_of_descriptors), axis=0
        )
        candidates_extra_scores = jax.tree.map(
            lambda x, y: jnp.concatenate((x, y), axis=0),
            self.extra_scores,
            filtered_batch_of_extra_scores,
        )

        # Compute dominated novelty
        _, dominated_novelty = _novelty_and_dominated_novelty(
            fitness=candidates_fitnesses[:, 0],
            descriptor=candidates_descriptors,
            novelty_k=self.k,
            dominated_novelty_k=self.k,
        )

        # Use dominated novelty as meta-fitness, invalid individuals get -inf
        valid = candidates_fitnesses[:, 0] != -jnp.inf
        meta_fitness = jnp.where(valid, dominated_novelty, -jnp.inf)

        # Select survivors
        indices = jnp.argsort(meta_fitness)[::-1]
        survivor_indices = indices[: self.size]

        new_genotypes = jax.tree.map(
            lambda x: x[survivor_indices], candidates_genotypes
        )
        new_fitnesses = candidates_fitnesses[survivor_indices]
        new_descriptors = candidates_descriptors[survivor_indices]
        new_extra_scores = jax.tree.map(
            lambda x: x[survivor_indices], candidates_extra_scores
        )

        return self.replace(  # type: ignore
            genotypes=new_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            extra_scores=new_extra_scores,
        )

    @classmethod
    def init(  # type: ignore
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        population_size: int,
        k: int,
        *args,
        extra_scores: Optional[ExtraScores] = None,
        keys_extra_scores: Tuple[str, ...] = (),
        **kwargs,
    ) -> DominatedNoveltyRepertoire:
        """Initialize the repertoire and add the first batch.

        Args:
            genotypes: first batch of genotypes (batch_size, ...)
            fitnesses: fitnesses of shape (batch_size, fitness_dim)
            descriptors: descriptors of shape (batch_size, num_descriptors)
            population_size: maximum number of individuals kept
            k: number of neighbors for novelty metrics
            extra_scores: extra scores of the first batch
            keys_extra_scores: keys of extra scores to store
        """

        if extra_scores is None:
            extra_scores = {}

        # retrieve one genotype and one extra score prototype
        first_genotype = jax.tree.map(lambda x: x[0], genotypes)
        first_extra_scores = jax.tree.map(lambda x: x[0], extra_scores)

        # create a repertoire with default values
        repertoire = cls.init_default(
            genotype=first_genotype,
            descriptor_dim=descriptors.shape[-1],
            population_size=population_size,
            one_extra_score=first_extra_scores,
            keys_extra_scores=keys_extra_scores,
            k=k,
        )

        # add initial population to the repertoire
        return repertoire.add(  # type: ignore
            genotypes, descriptors, fitnesses, extra_scores
        )

    @classmethod
    def init_default(
        cls,
        genotype: Genotype,
        descriptor_dim: int,
        population_size: int,
        one_extra_score: Optional[ExtraScores] = None,
        keys_extra_scores: Tuple[str, ...] = (),
        k: int = 15,
    ) -> DominatedNoveltyRepertoire:
        """Create a DNS repertoire with default values.

        Args:
            genotype: a representative genotype PyTree (leaf shapes define storage).
            descriptor_dim: number of descriptor dimensions.
            population_size: maximum number of individuals kept.
            one_extra_score: a representative extra score PyTree to size buffers.
            keys_extra_scores: keys of extra scores to store in the repertoire.
            k: number of neighbors for novelty metrics.

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

        # default fitness is -inf
        default_fitnesses = -jnp.inf * jnp.ones(shape=(population_size, 1))

        # default genotypes is all zeros
        default_genotypes = jax.tree.map(
            lambda x: jnp.zeros(shape=(population_size,) + x.shape, dtype=x.dtype),
            genotype,
        )

        # default descriptors is NaN (uninitialized)
        default_descriptors = jnp.full(
            shape=(population_size, descriptor_dim), fill_value=jnp.nan
        )

        # default extra scores buffers
        default_extra_scores = jax.tree.map(
            lambda x: jnp.zeros(shape=(population_size,) + x.shape, dtype=x.dtype),
            one_extra_score,
        )

        return cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            extra_scores=default_extra_scores,
            keys_extra_scores=keys_extra_scores,
            k=k,
        )
