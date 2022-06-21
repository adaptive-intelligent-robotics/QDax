"""Defines a repertoire for simple genetic algorithms."""

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

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
        return len(self.genotypes)

    def save(self, path: str = "./") -> None:
        """Saves the repertoire.

        Args:
            path: place to store the files. Defaults to "./".
        """
        jnp.save(path + "genotypes.npy", self.genotypes)
        jnp.save(path + "scores.npy", self.fitnesses)

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
        random_key, subkey = jax.random.split(random_key)
        mask = self.fitnesses != -jnp.inf
        p = jnp.any(mask, axis=-1) / jnp.sum(jnp.any(mask, axis=-1))

        # sample
        genotype_sample = jax.random.choice(
            key=subkey, a=self.genotypes, p=p, shape=(num_samples,), replace=False
        )

        return genotype_sample, random_key

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
        candidates = jnp.concatenate((self.genotypes, batch_of_genotypes))
        candidates_fitnesses = jnp.concatenate((self.fitnesses, batch_of_fitnesses))

        # sort by fitnesses
        indices = jnp.argsort(candidates_fitnesses)

        # keep only the best ones
        new_repertoire = self.replace(
            genotypes=candidates[indices], fitnesses=candidates_fitnesses[indices]
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

        default_fitnesses = -jnp.inf * jnp.ones(
            shape=(population_size, fitnesses.shape[-1])
        )
        default_genotypes = jnp.zeros(shape=(population_size, genotypes.shape[-1]))

        repertoire = cls(genotypes=default_genotypes, fitnesses=default_fitnesses)

        new_repertoire = repertoire.add(genotypes, fitnesses)

        return new_repertoire  # type: ignore
