"""
Definition of CMAES class, containing main functions necessary to build
a CMA optimization script. Link to the paper: https://arxiv.org/abs/1604.00772
"""
import functools
from typing import Callable, Optional, Tuple

import flax
import jax
import jax.numpy as jnp

from qdax.types import Fitness, Genotype, Mask, RNGKey


class CMAESState(flax.struct.PyTreeNode):
    """Describe a state of the Covariance matrix adaptation evolution strategy
    (CMA-ES) algorithm.

    Args:
        mean: mean of the gaussian distribution used to generate solutions
        cov_matrix: covariance matrix of the gaussian distribution used to
            generate solutions.
        num_updates: number of updates made by the CMAES optimizer since the
            beginning of the process.
        sigma: the step size of the optimization steps
    """

    mean: jnp.ndarray
    cov_matrix: jnp.ndarray
    num_updates: int
    sigma: float
    p_c: jnp.ndarray
    p_s: jnp.ndarray
    eigen_updates: int
    eigenvalues: jnp.ndarray
    invsqrt_cov: jnp.ndarray


class CMAES:
    """
    Class to run the CMA-ES algorithm.
    """

    def __init__(
        self,
        population_size: int,
        search_dim: int,
        fitness_function: Callable[[Genotype], Fitness],
        num_best: Optional[int] = None,
        init_sigma: float = 1e-3,
        mean_init: Optional[jnp.ndarray] = None,
        bias_weights: bool = True,
        delay_eigen_decomposition: bool = False,
    ):
        self._population_size = population_size
        self._search_dim = search_dim
        self._fitness_function = fitness_function
        self._init_sigma = init_sigma

        # Default values if values are not provided
        if num_best is None:
            self._num_best = population_size // 2
        else:
            self._num_best = num_best

        if mean_init is None:
            self._mean_init = jnp.zeros(shape=(search_dim,))
        else:
            self._mean_init = mean_init

        # weights parameters
        if bias_weights:
            self._weights = jnp.log(
                (self._num_best + 0.5) / jnp.arange(start=1, stop=(self._num_best + 1))
            )
        else:
            self._weights = jnp.ones(self._num_best)

        # scale weights
        self._weights = self._weights / (self._weights.sum())
        self._parents_eff = 1 / (self._weights**2).sum()

        # adaptation  parameters
        self._c_s = (self._parents_eff + 2) / (self._search_dim + self._parents_eff + 5)
        self._c_c = (4 + self._parents_eff / self._search_dim) / (
            self._search_dim + 4 + 2 * self._parents_eff / self._search_dim
        )

        # learning rate for rank-1 update of C
        self._c_1 = 2 / (self._parents_eff + (self._search_dim + jnp.sqrt(2)) ** 2)

        # learning rate for rank-(num best) updates
        tmp = 2 * (self._parents_eff - 2 + 1 / self._parents_eff)
        self._c_cov = min(
            1 - self._c_1, tmp / (self._parents_eff + (self._search_dim + 2) ** 2)
        )

        # damping for sigma
        self._d_s = (
            1
            + 2 * max(0, jnp.sqrt((self._parents_eff - 1) / (self._search_dim + 1)) - 1)
            + self._c_s
        )
        self._chi = jnp.sqrt(self._search_dim) * (
            1 - 1 / (4 * self._search_dim) + 1 / (21 * self._search_dim**2)
        )

        # threshold for new eigen decomposition
        # the sqrt is introduced in our implementation - hasn't been found elsewhere
        # reason for that was that the wikipedia heuristic was giving a period that
        # was getting way too big

        # TODO: should i simply use the one from pyribs???
        self._eigen_comput_period = 1
        if delay_eigen_decomposition:
            self._eigen_comput_period = (
                0.5 * jnp.sqrt(self._search_dim) / (10 * (self._c_1 + self._c_cov))
            )

    def init(self) -> CMAESState:
        """
        Init the CMA-ES algorithm.

        Returns:
            an initial state for the algorithm
        """

        cov_matrix = jnp.eye(self._search_dim)

        # cov is already diag
        invsqrt_cov = jnp.diag(1 / jnp.sqrt(jnp.diag(cov_matrix)))

        return CMAESState(
            mean=self._mean_init,
            cov_matrix=cov_matrix,
            sigma=self._init_sigma,
            num_updates=0,
            p_c=jnp.zeros(shape=(self._search_dim,)),
            p_s=jnp.zeros(shape=(self._search_dim,)),
            eigen_updates=0,
            eigenvalues=jnp.ones(shape=(self._search_dim,)),
            invsqrt_cov=invsqrt_cov,
        )

    @functools.partial(jax.jit, static_argnames=("self",))
    def sample(
        self, cmaes_state: CMAESState, random_key: RNGKey
    ) -> Tuple[Genotype, RNGKey]:
        """
        Sample a population.

        Args:
            cmaes_state: current state of the algorithm
            random_key: jax random key

        Returns:
            A tuple that contains a batch of population size genotypes and
            a new random key.
        """
        random_key, subkey = jax.random.split(random_key)
        samples = jax.random.multivariate_normal(
            subkey,
            shape=(self._population_size,),
            mean=cmaes_state.mean,
            cov=(cmaes_state.sigma**2) * cmaes_state.cov_matrix,
        )
        return samples, random_key

    @functools.partial(jax.jit, static_argnames=("self",))
    def update_state(
        self,
        cmaes_state: CMAESState,
        sorted_candidates: Genotype,
    ) -> CMAESState:
        return self._update_state(  # type: ignore
            cmaes_state=cmaes_state,
            sorted_candidates=sorted_candidates,
            weights=self._weights,
        )

    @functools.partial(jax.jit, static_argnames=("self",))
    def update_state_with_mask(
        self, cmaes_state: CMAESState, sorted_candidates: Genotype, mask: Mask
    ) -> CMAESState:

        weights = jnp.multiply(self._weights, mask)
        weights = weights / (weights.sum())

        return self._update_state(  # type: ignore
            cmaes_state=cmaes_state,
            sorted_candidates=sorted_candidates,
            weights=weights,
        )

    @functools.partial(jax.jit, static_argnames=("self",))
    def _update_state(
        self,
        cmaes_state: CMAESState,
        sorted_candidates: Genotype,
        weights: jnp.ndarray,
    ) -> CMAESState:

        """
        Updates the state when candidates have already been sorted and selected.

        Args:
            cmaes_state: current state of the algorithm
            sorted_candidates: a batch of sorted and selected genotypes

        Returns:
            An updated algorithm state
        """

        # retrieve elements from the current state
        p_c = cmaes_state.p_c
        p_s = cmaes_state.p_s
        sigma = cmaes_state.sigma
        num_updates = cmaes_state.num_updates
        cov = cmaes_state.cov_matrix
        mean = cmaes_state.mean

        eigen_updates = cmaes_state.eigen_updates
        eigenvalues = cmaes_state.eigenvalues
        invsqrt_cov = cmaes_state.invsqrt_cov

        # update mean by recombination
        old_mean = mean
        mean = weights @ sorted_candidates

        def update_eigen(
            operand: Tuple[jnp.ndarray, int]
        ) -> Tuple[int, jnp.ndarray, jnp.ndarray]:

            # unpack data
            cov, num_updates = operand

            # enfore symmetry - did not change anything
            cov = jnp.triu(cov) + jnp.triu(cov, 1).T

            # get eigen decomposition: eigenvalues, eigenvectors
            eig, u = jnp.linalg.eigh(cov)

            # compute new invsqrt
            invsqrt = u @ jnp.diag(1 / jnp.sqrt(eig)) @ u.T

            # update the eigen value decomposition tracker
            eigen_updates = num_updates

            return eigen_updates, eig, invsqrt

        # condition for recomputing the eig decomposition
        eigen_condition = (num_updates - eigen_updates) >= self._eigen_comput_period

        # decomposition of cov
        eigen_updates, eigenvalues, invsqrt = jax.lax.cond(
            eigen_condition,
            update_eigen,
            lambda _: (eigen_updates, eigenvalues, invsqrt_cov),
            operand=(cov, num_updates),
        )

        z = (1 / sigma) * (mean - old_mean)
        z_w = invsqrt @ z

        # update evolution paths - cumulation
        p_s = (1 - self._c_s) * p_s + jnp.sqrt(
            self._c_s * (2 - self._c_s) * self._parents_eff
        ) * z_w

        tmp_1 = jnp.linalg.norm(p_s) / jnp.sqrt(
            1 - (1 - self._c_s) ** (2 * num_updates)
        ) <= self._chi * (1.4 + 2 / (self._search_dim + 1))

        p_c = (1 - self._c_c) * p_c + tmp_1 * jnp.sqrt(
            self._c_c * (2 - self._c_c) * self._parents_eff
        ) * z

        # update covariance matrix
        pp_c = jnp.expand_dims(p_c, axis=1)

        coeff_tmp = (sorted_candidates - old_mean) / sigma
        cov_rank = coeff_tmp.T @ jnp.diag(weights.squeeze()) @ coeff_tmp

        cov = (
            (1 - self._c_cov - self._c_1) * cov
            + self._c_1
            * (pp_c @ pp_c.T + (1 - tmp_1) * self._c_c * (2 - self._c_c) * cov)
            + self._c_cov * cov_rank
        )

        # update step size
        sigma = sigma * jnp.exp(
            (self._c_s / self._d_s) * (jnp.linalg.norm(p_s) / self._chi - 1)
        )

        cmaes_state = CMAESState(
            mean=mean,
            cov_matrix=cov,
            sigma=sigma,
            num_updates=num_updates + 1,
            p_c=p_c,
            p_s=p_s,
            eigen_updates=eigen_updates,
            eigenvalues=eigenvalues,
            invsqrt_cov=invsqrt,
        )

        return cmaes_state

    @functools.partial(jax.jit, static_argnames=("self",))
    def update(self, cmaes_state: CMAESState, samples: Genotype) -> CMAESState:
        """
        Updates the distribution.

        Args:
            cmaes_state: current state of the algorithm
            samples: a batch of genotypes

        Returns:
            an updated algorithm state
        """

        fitnesses = -self._fitness_function(samples)
        idx_sorted = jnp.argsort(fitnesses)
        sorted_candidates = samples[idx_sorted[: self._num_best]]

        new_state = self.update_state(cmaes_state, sorted_candidates)

        return new_state  # type: ignore

    @functools.partial(jax.jit, static_argnames=("self",))
    def stop_condition(self, cmaes_state: CMAESState) -> bool:
        """Inspired from pyribs implementation.

        TODO: should we add the third condition implemented in pyribs?
        """
        # NaN appear because of float precision is reached
        nan_condition = jnp.sum(jnp.isnan(cmaes_state.eigenvalues)) > 0

        eig_dispersion = jnp.max(cmaes_state.eigenvalues) / jnp.min(
            cmaes_state.eigenvalues
        )
        first_condition = eig_dispersion > 1e14

        area = cmaes_state.sigma * jnp.sqrt(jnp.max(cmaes_state.eigenvalues))
        second_condition = area < 1e-11

        third_condition = jnp.max(cmaes_state.eigenvalues) < 1e-6  # 1e-9
        fourth_condition = jnp.min(cmaes_state.eigenvalues) > 1e6

        return (  # type: ignore
            nan_condition
            + first_condition
            + second_condition
            + third_condition
            + fourth_condition
        )
