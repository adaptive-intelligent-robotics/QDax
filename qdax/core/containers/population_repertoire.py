from __future__ import annotations

from functools import partial
from typing import Tuple

import flax
import jax
import jax.numpy as jnp

from qdax.types import Fitness, Genotype, RNGKey


class PopulationRepertoire(flax.struct.PyTreeNode):
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
    ) -> PopulationRepertoire:
        raise NotImplementedError

    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        population_size: int,
    ) -> PopulationRepertoire:

        # avoid the condition by doing the usual
        # default values + addition

        num_add = population_size - genotypes.shape[0]
        if num_add > 0:
            genotypes = jnp.concatenate(
                (genotypes, jnp.zeros((num_add, genotypes.shape[1]))), axis=0
            )
            fitnesses = jnp.concatenate(
                (fitnesses, -jnp.ones((num_add, fitnesses.shape[1])) * jnp.inf),
                axis=0,
            )

        new_repertoire = cls(genotypes=genotypes, fitnesses=fitnesses)

        return new_repertoire
