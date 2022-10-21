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
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Gradient, RNGKey


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
    """

    theta: Genotype
    theta_grads: Gradient
    random_key: RNGKey
    cmaes_state: CMAESState


class CMAMEGAEmitter(Emitter):
    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        batch_size: int,
        learning_rate: float,
        num_descriptors: int,
        sigma_g: float,
        step_size: Optional[float] = None,
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
            sigma_g: standard deviation for the coefficients
            step_size: size of the steps used in CMAES updates
        """
        self._scoring_function = scoring_function
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._weights = jnp.expand_dims(
            jnp.log(batch_size + 0.5) - jnp.log(jnp.arange(1, batch_size + 1)), axis=-1
        )
        self._weights = self._weights / (self._weights.sum())

        if step_size is None:
            step_size = 1.0

        # define a CMAES instance
        self._cmaes = CMAES(
            population_size=batch_size,
            search_dim=num_descriptors + 1,
            # no need for fitness function in that specific case
            fitness_function=None,  # type: ignore
            num_best=batch_size,
            init_sigma=sigma_g,
            init_step_size=step_size,
            bias_weights=True,
        )

        self._cma_initial_state = self._cmaes.init()

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[CMAMEGAState, RNGKey]:
        """
        Initializes the CMA-MEGA emitter


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

        # return the initial state
        random_key, subkey = jax.random.split(random_key)
        return (
            CMAMEGAState(
                theta=theta,
                theta_grads=theta_grads,
                cmaes_state=self._cma_initial_state,
                random_key=subkey,
            ),
            random_key,
        )

    @partial(jax.jit, static_argnames=("self", "batch_size"))
    def emit(
        self,
        repertoire: Optional[MapElitesRepertoire],
        emitter_state: CMAMEGAState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        Emits new individuals. Interestingly, this method does not directly modify
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
        coeffs, _ = self._cmaes.sample(
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
        improvements = fitnesses - repertoire.fitnesses[indices]

        sorted_indices = jnp.argsort(improvements)[::-1]

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
        reinitialize = jnp.all(improvements < 0)

        # re-sample
        random_theta, random_key = repertoire.sample(random_key, 1)

        # update
        theta = jnp.nan_to_num(theta) * (1 - reinitialize) + random_theta * reinitialize
        mean = self._cma_initial_state.mean * reinitialize + jnp.nan_to_num(
            cmaes_state.mean
        ) * (1 - reinitialize)
        cov = self._cma_initial_state.cov_matrix * reinitialize + jnp.nan_to_num(
            cmaes_state.cov_matrix
        ) * (1 - reinitialize)
        p_c = self._cma_initial_state.p_c * reinitialize + jnp.nan_to_num(
            cmaes_state.p_c
        ) * (1 - reinitialize)
        p_s = self._cma_initial_state.p_s * reinitialize + jnp.nan_to_num(
            cmaes_state.p_s
        ) * (1 - reinitialize)
        step_size = self._cma_initial_state.step_size * reinitialize + jnp.nan_to_num(
            cmaes_state.step_size
        ) * (1 - reinitialize)
        num_updates = 1 * reinitialize + cmaes_state.num_updates * (1 - reinitialize)

        # define new cmaes state
        cmaes_state = CMAESState(
            mean=mean,
            cov_matrix=cov,
            p_c=p_c,
            p_s=p_s,
            step_size=step_size,
            num_updates=num_updates,
        )

        # score theta
        _, _, extra_score, random_key = self._scoring_function(theta, random_key)

        # create new emitter state
        emitter_state = CMAMEGAState(
            theta=theta,
            theta_grads=extra_score["normalized_grads"],
            cmaes_state=cmaes_state,
            random_key=random_key,
        )

        return emitter_state
