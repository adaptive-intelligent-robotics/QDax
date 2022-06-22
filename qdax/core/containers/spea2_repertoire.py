from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.types import Fitness, Genotype


class SPEA2Repertoire(GARepertoire):
    """Repertoire used for the SPEA2 algorithm.

    Inherits from the GARepertoire. The data stored are the genotypes
    and there fitness. Several functions are inherited from GARepertoire,
    including size, save, sample and init.
    """

    _num_neighbours: int

    @partial(jax.jit, static_argnames=("self",))
    def _compute_strength_scores(self, batch_of_fitnesses: Fitness) -> jnp.ndarray:
        """Compute the strength scores (defined for a solution by the number of
        solutions dominating it).

        Args:
            batch_of_fitnesses: a batch of fitness vectors.

        Returns:
            Strength score of each solution corresponding to the fitnesses.
        """
        fitnesses = jnp.concatenate((self.fitnesses, batch_of_fitnesses), axis=0)
        dominates = jnp.all(
            (fitnesses - jnp.expand_dims(fitnesses, axis=1)) > 0, axis=-1
        )
        strength_scores = jnp.sum(dominates, axis=1)
        distance_matrix = jnp.sum(
            (fitnesses - jnp.expand_dims(fitnesses, axis=1)) ** 2, axis=-1
        )
        densities = jnp.sum(
            jnp.sort(distance_matrix, axis=1)[:, : self._num_neighbours + 1], axis=1
        )

        strength_scores = strength_scores + 1 / (1 + densities)
        strength_scores = jnp.nan_to_num(strength_scores, nan=self.size + 2)

        return strength_scores

    @partial(jax.jit, static_argnames=("self",))
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_fitnesses: Fitness,
    ) -> SPEA2Repertoire:
        """Updates the population with the new solutions.

        Args:
            batch_of_genotypes: genotypes of the new individuals that are
                considered to be added to the population.
            batch_of_fitnesses: their corresponding fitnesses.

        Returns:
            Updated repertoire.
        """
        # All the candidates
        candidates = jax.tree_map(
            lambda x, y: jnp.concatenate((x, y), axis=0),
            self.genotypes,
            batch_of_genotypes,
        )

        candidates_fitnesses = jnp.concatenate((self.fitnesses, batch_of_fitnesses))

        # Track front
        strength_scores = self._compute_strength_scores(
            self.genotypes, batch_of_fitnesses, self._num_neighbours
        )
        indices = jnp.argsort(strength_scores)[: self.size]
        new_candidates = jax.tree_map(lambda x: x[indices], candidates)
        new_fitnesses = candidates_fitnesses[indices]

        return SPEA2Repertoire(genotypes=new_candidates, fitnesses=new_fitnesses)
