"""
Definition of CMAES class, containing main functions necessary to build
a CMA optimization script. Link to the paper: https://arxiv.org/abs/1604.00772
"""
import functools
from typing import Callable, Optional, Tuple

import flax
import jax
import jax.numpy as jnp

from qdax.types import Fitness, Genotype, RNGKey


class CMAESState(flax.struct.PyTreeNode):
    """
    Describe a state of the Covariance matrix adaptation evolution strategy
    (CMA-ES) algorithm.
    """

    mean: jnp.ndarray
    cov_matrix: jnp.ndarray
    num_updates: int
    step_size: float
    p_c: jnp.ndarray
    p_s: jnp.ndarray


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
        weight_decay: float = 0.01,
        init_sigma: float = 1e-3,
        mean_init: Optional[jnp.ndarray] = None,
        bias_weights: bool = True,
        init_step_size: float = 1e-3,
    ):
        self._population_size = population_size
        self._weight_decay = weight_decay
        self._search_dim = search_dim
        self._fitness_function = fitness_function
        self._init_sigma = init_sigma
        self._init_step_size = init_step_size

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
                (self._num_best + 1) / jnp.arange(start=1, stop=(self._num_best + 1))
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
        tmp = self._parents_eff - 2 + 1 / self._parents_eff
        self._c_1 = 2 / (self._parents_eff + (self._search_dim + jnp.sqrt(2)) ** 2)
        self._c_cov = min(
            1 - self._c_1, tmp / (self._parents_eff + (self._search_dim + 2) ** 2)
        )
        self._d_s = (
            1
            + 2 * max(0, jnp.sqrt((self._parents_eff - 1) / (self._search_dim + 1) - 1))
            + self._c_s
        )
        self._chi = jnp.sqrt(self._search_dim) * (
            1 - 1 / (4 * self._search_dim) + 1 / (21 * self._search_dim**2)
        )

    def init(self) -> CMAESState:
        """
        Init the CMA-ES algorithm.

        Returns:
            an initial state for the algorithm
        """
        return CMAESState(
            mean=self._mean_init,
            cov_matrix=self._init_sigma * jnp.eye(self._search_dim),
            step_size=self._init_step_size,
            num_updates=1,
            p_c=jnp.zeros(shape=(self._search_dim,)),
            p_s=jnp.zeros(shape=(self._search_dim,)),
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
            cov=cmaes_state.cov_matrix,
        )
        return samples, random_key

    @functools.partial(jax.jit, static_argnames=("self",))
    def update_state(
        self, cmaes_state: CMAESState, sorted_candidates: Genotype
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
        step_size = cmaes_state.step_size
        num_updates = cmaes_state.num_updates
        cov = cmaes_state.cov_matrix
        mean = cmaes_state.mean

        # update mean
        old_mean = mean
        mean = self._weights @ sorted_candidates
        z = 1 / step_size * (mean - old_mean).T
        eig, u = jnp.linalg.eigh(cov)
        invsqrt = u @ jnp.diag(1 / jnp.sqrt(eig)) @ u.T
        z_w = invsqrt @ z

        # update evolution paths
        p_s = (1 - self._c_s) * p_s + jnp.sqrt(
            self._c_s * (2 - self._c_s) * self._parents_eff
        ) * z_w.squeeze()

        tmp_1 = jnp.linalg.norm(p_s) / jnp.sqrt(
            1 - (1 - self._c_s) ** (2 * num_updates)
        ) <= self._chi * (1.4 + 2 / (self._search_dim + 1))

        p_c = (1 - self._c_c) * p_c + 1 * jnp.sqrt(
            self._c_c * (2 - self._c_c) * self._parents_eff
        ) * z.squeeze()

        # update covariance matrix
        pp_c = jnp.expand_dims(p_c, axis=1)
        coeff_tmp = 1 / step_size * (sorted_candidates - mean)
        cov_rank = coeff_tmp.T @ jnp.diag(self._weights.squeeze()) @ coeff_tmp

        cov = (
            (1 - self._c_cov - self._c_1) * cov
            + self._c_1
            * (pp_c @ pp_c.T + (1 - tmp_1) * self._c_c * (2 - self._c_c) * cov)
            + self._c_cov * cov_rank
        )

        # update step size
        step_size = step_size * jnp.exp(
            (self._c_s / self._d_s) * (jnp.linalg.norm(p_s) / self._chi - 1)
        )

        cmaes_state = CMAESState(
            mean=mean,
            cov_matrix=cov,
            step_size=step_size,
            num_updates=num_updates + 1,
            p_c=p_c,
            p_s=p_s,
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
