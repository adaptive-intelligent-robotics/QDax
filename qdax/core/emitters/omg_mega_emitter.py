from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class OMGMEGAEmitterState(EmitterState):
    """
    Emitter state for the CMA-MEGA emitter.

    Args:
        gradients_repertoire: MapElites repertoire containing the gradients
            of the indivuals.
    """

    gradients_repertoire: MapElitesRepertoire


class OMGMEGAEmitter(Emitter):
    """
    Class for the emitter of OMG Mega from "Differentiable Quality Diversity" by
    Fontaine et al.

    NOTE: in order to implement this emitter while staying in the MAPElites
    framework, we had to make two temporary design choices:
    - in the emit function, we use the same random key to sample from the
    genotypes and gradients repertoire, in order to get the gradients that
    correspond to the right genotypes. Although acceptable, this is definitely
    not the best coding practice and we would prefer to get rid of this in a
    future version. A solution that we are discussing with the development team
    is to decompose the sampling function of the repertoire into two phases: one
    sampling the indices to be sampled, the other one retrieving the corresponding
    elements. This would enable to reuse the indices instead of doing this double
    sampling.
    - in the state_update, we have to insert the gradients in the gradients
    repertoire in the same way the individuals were inserted. Once again, this is
    slightly unoptimal because the same addition mecanism has to be computed two
    times. One solution that we are discussing and that is very similar to the first
    solution discussed above, would be to decompose the addition mecanism in two
    phases: one outputing the indices at which individuals will be added, and then
    the actual insertion step. This would enable to re-use the same indices to add
    the gradients instead of having to recompute them.

    The two design choices seem acceptable and enable to have OMG MEGA compatible
    with the current implementation of the MAPElites and MAPElitesRepertoire classes.

    Our suggested solutions seem quite simple and are likely to be useful for other
    variants implementation. They will be further discussed with the development team
    and potentially added in a future version of the package.
    """

    def __init__(
        self,
        batch_size: int,
        sigma_g: float,
        num_descriptors: int,
        centroids: Centroid,
    ):
        """Creates an instance of the OMGMEGAEmitter class.

        Args:
            batch_size: number of solutions sampled at each iteration
            sigma_g: CAUTION - square of the standard deviation for the coefficients.
                This notation can be misleading as, although it's called sigma, it
                refers to the variance and not the standard deviation.
            num_descriptors: number of descriptors
            centroids: centroids used to create the repertoire of solutions.
                This will be used to create the repertoire of gradients.
        """
        # set the mean of the coeff distribution to zero
        self._mu = jnp.zeros(num_descriptors + 1)

        # set the cov matrix to sigma * I
        self._sigma = jnp.eye(num_descriptors + 1) * sigma_g

        # define other parameters of the distribution
        self._batch_size = batch_size
        self._centroids = centroids
        self._num_descriptors = num_descriptors

    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[OMGMEGAEmitterState, RNGKey]:
        """Initialises the state of the emitter. Creates an empty repertoire
        that will later contain the gradients of the individuals.

        Args:
            init_genotypes: The genotypes of the initial population.
            random_key: a random key to handle stochastic operations.

        Returns:
            The initial emitter state.
        """
        # retrieve one genotype from the population
        first_genotype = jax.tree_util.tree_map(lambda x: x[0], init_genotypes)

        # add a dimension of size num descriptors + 1
        gradient_genotype = jax.tree_util.tree_map(
            lambda x: jnp.repeat(
                jnp.expand_dims(x, axis=-1), repeats=self._num_descriptors + 1, axis=-1
            ),
            first_genotype,
        )

        # create the gradients repertoire
        gradients_repertoire = MapElitesRepertoire.init_default(
            genotype=gradient_genotype, centroids=self._centroids
        )

        return (
            OMGMEGAEmitterState(gradients_repertoire=gradients_repertoire),
            random_key,
        )

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: OMGMEGAEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        OMG emitter function that samples elements in the repertoire and does a gradient
        update with random coefficients to create new candidates.

        Args:
            repertoire: current repertoire
            emitter_state: current emitter state, contains the gradients
            random_key: random key

        Returns:
            new_genotypes: new candidates to be added to the grid
            random_key: updated random key
        """
        # sample genotypes
        (
            genotypes,
            _,
        ) = repertoire.sample(random_key, num_samples=self._batch_size)

        # sample gradients - use the same random key for sampling
        # See class docstrings for discussion about this choice
        gradients, random_key = emitter_state.gradients_repertoire.sample(
            random_key, num_samples=self._batch_size
        )

        fitness_gradients = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x[:, :, 0], axis=-1), gradients
        )
        descriptors_gradients = jax.tree_util.tree_map(lambda x: x[:, :, 1:], gradients)

        # Normalize the gradients
        norm_fitness_gradients = jnp.linalg.norm(
            fitness_gradients, axis=1, keepdims=True
        )

        fitness_gradients = fitness_gradients / norm_fitness_gradients

        norm_descriptors_gradients = jnp.linalg.norm(
            descriptors_gradients, axis=1, keepdims=True
        )
        descriptors_gradients = descriptors_gradients / norm_descriptors_gradients

        # Draw random coefficients
        random_key, subkey = jax.random.split(random_key)
        coeffs = jax.random.multivariate_normal(
            subkey,
            shape=(self._batch_size,),
            mean=self._mu,
            cov=self._sigma,
        )
        coeffs = coeffs.at[:, 0].set(jnp.abs(coeffs[:, 0]))
        grads = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate((x, y), axis=-1),
            fitness_gradients,
            descriptors_gradients,
        )
        update_grad = jnp.sum(jax.vmap(lambda x, y: x * y)(coeffs, grads), axis=-1)

        # update the genotypes
        new_genotypes = jax.tree_util.tree_map(
            lambda x, y: x + y, genotypes, update_grad
        )

        return new_genotypes, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: OMGMEGAEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> OMGMEGAEmitterState:
        """Update the gradients repertoire to have the right gradients.

        NOTE: see discussion in the class docstrings to see how this could
        be improved.

        Args:
            emitter_state: current emitter state
            repertoire: the current genotypes repertoire
            genotypes: the genotypes of the batch of emitted offspring.
            fitnesses: the fitnesses of the batch of emitted offspring.
            descriptors: the descriptors of the emitted offspring.
            extra_scores: a dictionary with other values outputted by the
                scoring function.

        Returns:
            The modified emitter state.
        """

        # get gradients out of the extra scores
        assert "gradients" in extra_scores.keys(), "Missing gradients or wrong key"
        gradients = extra_scores["gradients"]

        # update the gradients repertoire
        gradients_repertoire = emitter_state.gradients_repertoire.add(
            gradients,
            descriptors,
            fitnesses,
            extra_scores,
        )

        return emitter_state.replace(  # type: ignore
            gradients_repertoire=gradients_repertoire
        )

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._batch_size
