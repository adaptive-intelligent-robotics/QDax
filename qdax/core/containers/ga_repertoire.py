"""Defines a repertoire for simple genetic algorithms."""

from __future__ import annotations

from typing import Optional, Tuple

import flax
import jax
import jax.numpy as jnp

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.repertoire_selectors.selector import GARepertoireT, Selector
from qdax.core.emitters.repertoire_selectors.uniform_selector import UniformSelector
from qdax.custom_types import ExtraScores, Fitness, Genotype, RNGKey


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
        extra_scores: extra scores resulting from the evaluation of the genotypes
        keys_extra_scores: keys of the extra scores to store in the repertoire
    """

    genotypes: Genotype
    fitnesses: Fitness
    extra_scores: ExtraScores
    keys_extra_scores: Tuple[str, ...] = flax.struct.field(
        pytree_node=False,
    )

    @property
    def size(self) -> int:
        """Gives the size of the population."""
        first_leaf = jax.tree.leaves(self.genotypes)[0]
        return int(first_leaf.shape[0])

    def select(
        self,
        key: RNGKey,
        num_samples: int,
        selector: Optional[Selector[GARepertoireT]] = None,
    ) -> GARepertoireT:
        if selector is None:
            selector = UniformSelector(select_with_replacement=False)
        repertoire = selector.select(self, key, num_samples)
        return repertoire

    def filter_extra_scores(self, extra_scores: ExtraScores) -> ExtraScores:
        filtered_extra_scores = {
            key: value
            for key, value in extra_scores.items()
            if key in self.keys_extra_scores
        }
        return filtered_extra_scores

    def add(  # type: ignore
        self,
        batch_of_genotypes: Genotype,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
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
        if batch_of_extra_scores is None:
            batch_of_extra_scores = {}

        filtered_batch_of_extra_scores = self.filter_extra_scores(batch_of_extra_scores)

        # gather individuals and fitnesses
        candidates = jax.tree.map(
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
        new_candidates = jax.tree.map(lambda x: x[survivor_indices], candidates)
        new_extra_scores = jax.tree.map(
            lambda x: x[survivor_indices], filtered_batch_of_extra_scores
        )
        new_repertoire = self.replace(
            genotypes=new_candidates,
            fitnesses=candidates_fitnesses[survivor_indices],
            extra_scores=new_extra_scores,
        )

        return new_repertoire  # type: ignore

    @classmethod
    def init(  # type: ignore
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        population_size: int,
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
        )

        new_repertoire = repertoire.add(genotypes, fitnesses, extra_scores)

        return new_repertoire  # type: ignore
