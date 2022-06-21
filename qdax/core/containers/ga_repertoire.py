from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.repertoire import Repertoire
from qdax.types import Fitness, Genotype, RNGKey


class GARepertoire(Repertoire):
    genotypes: Genotype
    fitnesses: Fitness

    @property
    def size(self) -> int:
        return len(self.genotypes)

    def save(self, path: str = "./") -> None:
        jnp.save(path + "genotypes.npy", self.genotypes)
        jnp.save(path + "scores.npy", self.fitnesses)

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:

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

        default_fitnesses = -jnp.inf * jnp.ones(
            shape=(population_size, fitnesses.shape[-1])
        )
        default_genotypes = jnp.zeros(shape=(population_size, genotypes.shape[-1]))

        repertoire = cls(genotypes=default_genotypes, fitnesses=default_fitnesses)

        new_repertoire = repertoire.add(genotypes, fitnesses)

        return new_repertoire  # type: ignore
