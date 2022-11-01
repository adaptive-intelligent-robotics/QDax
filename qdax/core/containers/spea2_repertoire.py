from __future__ import annotations

import flax
import jax
import jax.numpy as jnp

from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.types import Fitness, Genotype


class SPEA2Repertoire(GARepertoire):
    """Repertoire used for the SPEA2 algorithm.

    Inherits from the GARepertoire. The data stored are the genotypes
    and there fitness. Several functions are inherited from GARepertoire,
    including size, save, sample.
    """

    num_neighbours: int = flax.struct.field(pytree_node=False)

    @jax.jit
    def _compute_strength_scores(self, batch_of_fitnesses: Fitness) -> jnp.ndarray:
        """Compute the strength scores (defined for a solution by the number of
        solutions dominating it plus the inverse of the density of solution in the
        fitness space).

        Args:
            batch_of_fitnesses: a batch of fitness vectors.

        Returns:
            Strength score of each solution corresponding to the fitnesses.
        """
        fitnesses = jnp.concatenate((self.fitnesses, batch_of_fitnesses), axis=0)
        # dominating solutions
        dominates = jnp.all(
            (fitnesses - jnp.expand_dims(fitnesses, axis=1)) > 0, axis=-1
        )
        strength_scores = jnp.sum(dominates, axis=1)

        # density
        distance_matrix = jnp.sum(
            (fitnesses - jnp.expand_dims(fitnesses, axis=1)) ** 2, axis=-1
        )
        densities = jnp.sum(
            jnp.sort(distance_matrix, axis=1)[:, : self.num_neighbours + 1], axis=1
        )

        # sum both terms
        strength_scores = strength_scores + 1 / (1 + densities)
        strength_scores = jnp.nan_to_num(strength_scores, nan=self.size + 2)

        return strength_scores

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_fitnesses: Fitness,
    ) -> SPEA2Repertoire:
        """Updates the population with the new solutions.

        To decide which individuals to keep, we count, for each solution,
        the number of solutions by which tey are dominated. We keep only
        the solutions that are the less dominated ones.

        Args:
            batch_of_genotypes: genotypes of the new individuals that are
                considered to be added to the population.
            batch_of_fitnesses: their corresponding fitnesses.

        Returns:
            Updated repertoire.
        """
        # All the candidates
        candidates = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate((x, y), axis=0),
            self.genotypes,
            batch_of_genotypes,
        )

        candidates_fitnesses = jnp.concatenate((self.fitnesses, batch_of_fitnesses))

        # compute strength score for all solutions
        strength_scores = self._compute_strength_scores(batch_of_fitnesses)

        # sort the strengths (the smaller the better (sic, respect paper's notation))
        indices = jnp.argsort(strength_scores)[: self.size]

        # keep the survivors
        new_candidates = jax.tree_util.tree_map(lambda x: x[indices], candidates)
        new_fitnesses = candidates_fitnesses[indices]

        new_repertoire = self.replace(genotypes=new_candidates, fitnesses=new_fitnesses)

        return new_repertoire  # type: ignore

    @classmethod
    def init(  # type: ignore
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        population_size: int,
        num_neighbours: int,
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
        repertoire = cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            num_neighbours=num_neighbours,
        )

        new_repertoire = repertoire.add(genotypes, fitnesses)

        return new_repertoire  # type: ignore
