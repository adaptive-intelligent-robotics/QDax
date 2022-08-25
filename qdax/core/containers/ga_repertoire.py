"""Defines a repertoire for simple genetic algorithms."""

from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from qdax.core.containers.repertoire import Repertoire
from qdax.types import Fitness, Genotype, RNGKey


class GARepertoire(Repertoire):
    """Class for a simple repertoire for a simple genetic
    algorithm.

    Args:
        genotypes: a PyTree containing the genotypes of the
            individuals in the population. Each leaf has the
            shape (population_size, num_features).
        fitnesses: an array containing the fitness of the individuals
            in the population. With shape (population_size, fitness_dim).
            The implementation of GARepertoire was thought for the case
            where fitness_dim equals 1 but the class can be herited and
            rules adapted for cases where fitness_dim is greater than 1.
    """

    genotypes: Genotype
    fitnesses: Fitness

    @property
    def size(self) -> int:
        """Gives the size of the population."""
        first_leaf = jax.tree_util.tree_leaves(self.genotypes)[0]
        return int(first_leaf.shape[0])

    def save(self, path: str = "./") -> None:
        """Saves the repertoire.

        Args:
            path: place to store the files. Defaults to "./".
        """

        def flatten_genotype(genotype: Genotype) -> jnp.ndarray:
            flatten_genotype, _ = ravel_pytree(genotype)
            return flatten_genotype

        # flatten all the genotypes
        flat_genotypes = jax.vmap(flatten_genotype)(self.genotypes)

        jnp.save(path + "genotypes.npy", flat_genotypes)
        jnp.save(path + "scores.npy", self.fitnesses)

    @classmethod
    def load(cls, reconstruction_fn: Callable, path: str = "./") -> GARepertoire:
        """Loads a GA Repertoire.

        Args:
            reconstruction_fn: Function to reconstruct a PyTree
                from a flat array.
            path: Path where the data is saved. Defaults to "./".

        Returns:
            A GA Repertoire.
        """

        flat_genotypes = jnp.load(path + "genotypes.npy")
        genotypes = jax.vmap(reconstruction_fn)(flat_genotypes)

        fitnesses = jnp.load(path + "fitnesses.npy")

        return cls(
            genotypes=genotypes,
            fitnesses=fitnesses,
        )

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:
        """Sample genotypes from the repertoire.

        Args:
            random_key: a random key to handle stochasticity.
            num_samples: the number of genotypes to sample.

        Returns:
            The sample of genotypes.
        """

        # prepare sampling probability
        mask = self.fitnesses != -jnp.inf
        p = jnp.any(mask, axis=-1) / jnp.sum(jnp.any(mask, axis=-1))

        # sample
        random_key, subkey = jax.random.split(random_key)
        samples = jax.tree_util.tree_map(
            lambda x: jax.random.choice(
                subkey, x, shape=(num_samples,), p=p, replace=False
            ),
            self.genotypes,
        )

        return samples, random_key

    @jax.jit
    def add(
        self, batch_of_genotypes: Genotype, batch_of_fitnesses: Fitness
    ) -> GARepertoire:
        """Implements the repertoire addition rules.

        Parents and offsprings are gathered and only the population_size
        bests are kept. The others are killed.

        Args:
            batch_of_genotypes: new genotypes that we try to add.
            batch_of_fitnesses: fitness of those new genotypes.

        Returns:
            The updated repertoire.
        """

        # gather individuals and fitnesses
        candidates = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate((x, y), axis=0),
            self.genotypes,
            batch_of_genotypes,
        )
        candidates_fitnesses = jnp.concatenate(
            (self.fitnesses, batch_of_fitnesses), axis=0
        )

        # sort by fitnesses
        indices = jnp.argsort(jnp.sum(candidates_fitnesses, axis=1))[::-1]

        # keep only the best ones
        survivor_indices = indices[: self.size]

        # keep only the best ones
        new_candidates = jax.tree_util.tree_map(
            lambda x: x[survivor_indices], candidates
        )

        new_repertoire = self.replace(
            genotypes=new_candidates, fitnesses=candidates_fitnesses[survivor_indices]
        )

        return new_repertoire  # type: ignore

    @classmethod
    def init(  # type: ignore
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        population_size: int,
    ) -> GARepertoire:
        """Initializes the repertoire.

        Start with default values and adds a first batch of genotypes
        to the repertoire.

        Args:
            genotypes: first batch of genotypes
            fitnesses: corresponding fitnesses
            population_size: size of the population we want to evolve

        Returns:
            An initial repertoire.
        """
        # create default fitnesses
        default_fitnesses = -jnp.inf * jnp.ones(
            shape=(population_size, fitnesses.shape[-1])
        )

        # create default genotypes
        default_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.zeros(shape=(population_size,) + x.shape[1:]), genotypes
        )

        # create an initial repertoire with those default values
        repertoire = cls(genotypes=default_genotypes, fitnesses=default_fitnesses)

        new_repertoire = repertoire.add(genotypes, fitnesses)

        return new_repertoire  # type: ignore
