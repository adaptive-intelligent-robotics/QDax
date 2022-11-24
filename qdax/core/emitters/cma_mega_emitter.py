from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.cmaes import CMAES, CMAESState
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Gradient,
    RNGKey,
)


class CMAMEGAState(EmitterState):
    """
    Emitter state for the CMA-MEGA emitter.

    Args:
        theta: current genotype from where candidates will be drawn.
        theta_grads: normalized fitness and descriptors gradients of theta.
        random_key: a random key to handle stochastic operations. Used for
            state update only, another key is used to emit. This might be
            subject to refactoring discussions in the future.
        cmaes_state: state of the underlying CMA-ES algorithm
        previous_fitnesses: store last fitnesses of the repertoire. Used to
            compute the improvment.
    """

    theta: Genotype
    theta_grads: Gradient
    random_key: RNGKey
    cmaes_state: CMAESState
    previous_fitnesses: Fitness


class CMAMEGAEmitter(Emitter):
    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        batch_size: int,
        learning_rate: float,
        num_descriptors: int,
        centroids: Centroid,
        sigma_g: float,
    ):
        """
        Class for the emitter of CMA Mega from "Differentiable Quality Diversity" by
        Fontaine et al.

        Args:
            scoring_function: a function to score individuals, outputing fitness,
                descriptors and extra scores. With this emitter, the extra score
                contains gradients and normalized gradients.
            batch_size: number of solutions sampled at each iteration
            learning_rate: rate at which the mean of the distribution is updated.
            num_descriptors: number of descriptors
            centroids: centroids of the repertoire used to store the genotypes
            sigma_g: standard deviation for the coefficients
        """

        self._scoring_function = scoring_function
        self._batch_size = batch_size
        self._learning_rate = learning_rate

        # weights used to update the gradient direction through a linear combination
        self._weights = jnp.expand_dims(
            jnp.log(batch_size + 0.5) - jnp.log(jnp.arange(1, batch_size + 1)), axis=-1
        )
        self._weights = self._weights / (self._weights.sum())

        # define a CMAES instance - used to update the coeffs
        self._cmaes = CMAES(
            population_size=batch_size,
            search_dim=num_descriptors + 1,
            # no need for fitness function in that specific case
            fitness_function=None,  # type: ignore
            num_best=batch_size,
            init_sigma=sigma_g,
            bias_weights=True,
            delay_eigen_decomposition=True,
        )

        self._centroids = centroids

        self._cma_initial_state = self._cmaes.init()

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[CMAMEGAState, RNGKey]:
        """
        Initializes the CMA-MEGA emitter.


        Args:
            init_genotypes: initial genotypes to add to the grid.
            random_key: a random key to handle stochastic operations.

        Returns:
            The initial state of the emitter.
        """

        # define init theta as 0
        theta = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x[:1, ...]),
            init_genotypes,
        )

        # score it
        _, _, extra_score, random_key = self._scoring_function(theta, random_key)
        theta_grads = extra_score["normalized_grads"]

        # Initialize repertoire with default values
        num_centroids = self._centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)

        # return the initial state
        random_key, subkey = jax.random.split(random_key)
        return (
            CMAMEGAState(
                theta=theta,
                theta_grads=theta_grads,
                random_key=subkey,
                cmaes_state=self._cma_initial_state,
                previous_fitnesses=default_fitnesses,
            ),
            random_key,
        )

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Optional[MapElitesRepertoire],
        emitter_state: CMAMEGAState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        Emits new individuals. Interestingly, this method does not directly modifies
        individuals from the repertoire but sample from a distribution. Hence the
        repertoire is not used in the emit function.

        Args:
            repertoire: a repertoire of genotypes (unused).
            emitter_state: the state of the CMA-MEGA emitter.
            random_key: a random key to handle random operations.

        Returns:
            New genotypes and a new random key.
        """

        # retrieve elements from the emitter state
        theta = jnp.nan_to_num(emitter_state.theta)
        cmaes_state = emitter_state.cmaes_state

        # get grads - remove nan and first dimension
        grads = jnp.nan_to_num(emitter_state.theta_grads.squeeze(axis=0))

        # Draw random coefficients - use the emitter state key
        coeffs, random_key = self._cmaes.sample(
            cmaes_state=cmaes_state, random_key=emitter_state.random_key
        )

        # make sure the fitness coefficient is positive
        coeffs = coeffs.at[:, 0].set(jnp.abs(coeffs[:, 0]))
        update_grad = coeffs @ grads.T

        # Compute new candidates
        new_thetas = jax.tree_util.tree_map(lambda x, y: x + y, theta, update_grad)

        return new_thetas, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: CMAMEGAState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Optional[EmitterState]:
        """
        Updates the CMA-MEGA emitter state.

        Note: in order to recover the coeffs that where used to sample the genotypes,
        we reuse the emitter state's random key in this function.

        Note: we use the update_state function from CMAES, a function that suppose
        that the candidates are already sorted. We do this because we have to sort
        them in this function anyway, in order to apply the right weights to the
        terms when update theta.

        Args:
            emitter_state: current emitter state
            repertoire: the current genotypes repertoire
            genotypes: the genotypes of the batch of emitted offspring (unused).
            fitnesses: the fitnesses of the batch of emitted offspring.
            descriptors: the descriptors of the emitted offspring.
            extra_scores: unused

        Returns:
            The updated emitter state.
        """

        # retrieve elements from the emitter state
        cmaes_state = emitter_state.cmaes_state
        theta = jnp.nan_to_num(emitter_state.theta)
        grads = jnp.nan_to_num(emitter_state.theta_grads[0])

        # Update the archive and compute the improvements
        indices = get_cells_indices(descriptors, repertoire.centroids)
        improvements = fitnesses - emitter_state.previous_fitnesses[indices]

        # condition for being a new cell
        condition = improvements == jnp.inf

        # criteria: fitness if new cell, improvement else
        ranking_criteria = jnp.where(condition, x=fitnesses, y=improvements)

        # make sure to have all the new cells first
        new_cell_offset = jnp.max(ranking_criteria) - jnp.min(ranking_criteria)

        ranking_criteria = jnp.where(
            condition, x=ranking_criteria + new_cell_offset, y=ranking_criteria
        )

        # sort indices according to the criteria
        sorted_indices = jnp.flip(jnp.argsort(ranking_criteria))

        # Draw the coeffs - reuse the emitter state key to get same coeffs
        coeffs, random_key = self._cmaes.sample(
            cmaes_state=cmaes_state, random_key=emitter_state.random_key
        )
        # make sure the fitness coeff is positive
        coeffs = coeffs.at[:, 0].set(jnp.abs(coeffs[:, 0]))

        # get the gradients that must be applied
        update_grad = coeffs @ grads.T

        # weight terms - based on improvement rank
        gradient_step = jnp.sum(self._weights[sorted_indices] * update_grad, axis=0)

        # update theta
        theta = jax.tree_util.tree_map(
            lambda x, y: x + self._learning_rate * y, theta, gradient_step
        )

        # Update CMA Parameters
        sorted_candidates = coeffs[sorted_indices]
        cmaes_state = self._cmaes.update_state(cmaes_state, sorted_candidates)

        # If no improvement draw randomly and re-initialize parameters
        reinitialize = jnp.all(improvements < 0) + self._cmaes.stop_condition(
            cmaes_state
        )

        # re-sample
        random_theta, random_key = repertoire.sample(random_key, 1)

        # update theta in case of reinit
        theta = jax.tree_util.tree_map(
            lambda x, y: jnp.where(reinitialize, x=x, y=y), random_theta, theta
        )

        # update cmaes state in case of reinit
        cmaes_state = jax.tree_util.tree_map(
            lambda x, y: jnp.where(reinitialize, x=x, y=y),
            self._cma_initial_state,
            cmaes_state,
        )

        # score theta
        _, _, extra_score, random_key = self._scoring_function(theta, random_key)

        # create new emitter state
        emitter_state = CMAMEGAState(
            theta=theta,
            theta_grads=extra_score["normalized_grads"],
            random_key=random_key,
            cmaes_state=cmaes_state,
            previous_fitnesses=repertoire.fitnesses,
        )

        return emitter_state

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._batch_size
