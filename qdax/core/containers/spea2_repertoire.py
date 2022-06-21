from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.types import Fitness, Genotype


class SPEA2Repertoire(GARepertoire):

    num_neighbours: int

    @partial(jax.jit, static_argnames=("self",))
    def compute_strength_scores(self, batch_of_fitnesses: Fitness) -> jnp.ndarray:
        """
        Compute the strength scores (defined for a solution by the number of solutions
        dominating it)
        """
        scores = jnp.concatenate((self.genotypes, batch_of_fitnesses), axis=0)
        dominates = jnp.all((scores - jnp.expand_dims(scores, axis=1)) > 0, axis=-1)
        strength_scores = jnp.sum(dominates, axis=1)
        distance_matrix = jnp.sum(
            (scores - jnp.expand_dims(scores, axis=1)) ** 2, axis=-1
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
        """
        Updates the population with the new solutions
        """
        # All the candidates
        candidates = jnp.concatenate((self.genotypes, batch_of_genotypes))
        candidate_scores = jnp.concatenate((self.fitnesses, batch_of_fitnesses))

        # Track front
        strength_scores = self._compute_strength_scores(
            self.solutions, batch_of_fitnesses, self._num_neighbours
        )
        indices = jnp.argsort(strength_scores)[: len(self.genotypes)]
        new_candidates = candidates[indices]
        new_scores = candidate_scores[indices]

        return SPEA2Repertoire(genotypes=new_candidates, fitnesses=new_scores)
