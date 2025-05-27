from __future__ import annotations

from typing import Optional, Tuple

import flax
import jax
import jax.numpy as jnp

from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.custom_types import ExtraScores, Fitness, Genotype


class SPEA2Repertoire(GARepertoire):
    """Repertoire used for the SPEA2 algorithm.

    Inherits from the GARepertoire. The data stored are the genotypes
    and there fitness. Several functions are inherited from GARepertoire,
    including size, sample.
    """

    num_neighbours: int = flax.struct.field(pytree_node=False)

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

    def add(  # type: ignore
        self,
        batch_of_genotypes: Genotype,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> SPEA2Repertoire:
        """Updates the population with the new solutions.

        To decide which individuals to keep, we count, for each solution,
        the number of solutions by which they are dominated. We keep only
        the solutions that are the less dominated ones.

        Args:
            batch_of_genotypes: genotypes of the new individuals that are
                considered to be added to the population.
            batch_of_fitnesses: their corresponding fitnesses.
            batch_of_extra_scores: extra scores of those new genotypes.

        Returns:
            Updated repertoire.
        """

        if batch_of_extra_scores is None:
            batch_of_extra_scores = {}

        # All the candidates
        candidates = jax.tree.map(
            lambda x, y: jnp.concatenate((x, y), axis=0),
            self.genotypes,
            batch_of_genotypes,
        )

        candidates_fitnesses = jnp.concatenate((self.fitnesses, batch_of_fitnesses))

        # compute strength score for all solutions
        strength_scores = self._compute_strength_scores(batch_of_fitnesses)

        # sort the strengths (the smaller the better (sic, respect paper's notation))
        indices = jnp.argsort(strength_scores)[: self.size]

        # keep only the survivors
        new_candidates = jax.tree.map(lambda x: x[indices], candidates)
        new_scores = candidates_fitnesses[indices]

        filtered_batch_of_extra_scores = self.filter_extra_scores(batch_of_extra_scores)
        new_extra_scores = jax.tree.map(
            lambda x: x[indices], filtered_batch_of_extra_scores
        )
        new_repertoire = self.replace(
            genotypes=new_candidates,
            fitnesses=new_scores,
            extra_scores=new_extra_scores,
        )

        return new_repertoire  # type: ignore

    @classmethod
    def init(  # type: ignore
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        population_size: int,
        num_neighbours: int,
        *args,
        extra_scores: Optional[ExtraScores] = None,
        keys_extra_scores: Tuple[str, ...] = (),
        **kwargs,
    ) -> GARepertoire:
        """Initializes the repertoire.

        Start with default values and adds a first batch of genotypes
        to the repertoire.

        Args:
            genotypes: first batch of genotypes
            fitnesses: corresponding fitnesses
            population_size: size of the population we want to evolve
            extra_scores: extra scores resulting from the evaluation of the genotypes
            keys_extra_scores: keys of the extra scores to store in the repertoire

        Returns:
            An initial repertoire.
        """

        if extra_scores is None:
            extra_scores = {}

        # create default fitnesses
        default_fitnesses = -jnp.inf * jnp.ones(
            shape=(population_size, fitnesses.shape[-1])
        )

        # create default genotypes
        default_genotypes = jax.tree.map(
            lambda x: jnp.zeros(shape=(population_size,) + x.shape[1:]), genotypes
        )

        # create default extra scores
        filtered_extra_scores = {
            key: value
            for key, value in extra_scores.items()
            if key in keys_extra_scores
        }

        default_extra_scores = jax.tree.map(
            lambda x: jnp.zeros(shape=(population_size,) + x.shape[1:]),
            filtered_extra_scores,
        )

        # create an initial repertoire with those default values
        repertoire = cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            extra_scores=default_extra_scores,
            keys_extra_scores=keys_extra_scores,
            num_neighbours=num_neighbours,
        )

        new_repertoire = repertoire.add(genotypes, fitnesses, extra_scores)

        return new_repertoire  # type: ignore
