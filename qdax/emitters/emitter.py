from typing import Any, Tuple

from qdax.algorithms.map_elites import MapElitesRepertoire
from qdax.types import EmitterState, Genotype, RNGKey


class Emitter:
    def emit_fn(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: EmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, EmitterState, RNGKey]:
        """Function used to emit a population of offspring by any possible
        mean. New population can be sampled from a distribution or obtained
        through mutations of individuals sampled from the repertoire.


        Args:
            repertoire: a repertoire of genotypes.
            emitter_state: the state of the emitter.
            random_key: a random key to handle random operations.

        Raises:
            NotImplementedError: this function needs to be overridden.

        Returns:
            A batch of offspring, a (potentially) new emitter state and
                a new random key.
        """
        raise NotImplementedError

    def state_update_fn(
        self, emitter_state: EmitterState, **kwargs: Any
    ) -> EmitterState:
        """This function gives an opportunity to update the emitter state
        after the genotypes have been scored.

        As a matter of fact, many emitter states needs informations from
        the evaluations of the genotypes in order to be updated, for instance:
        - CMA emitter: to update the rank of the covariance matrix
        - PGA emitter: to fill the replay buffer.

        This function does not need to be overridden. By default, it output
        the same emitter state.

        Args:
            emitter_state: current emitter state
            kwargs: contains information from the evaluation of the population
                of genotypes, like the fitness and the descriptors obtained, the
                transitions experienced during the evaluations. And potentially
                other elements outputted by the scoring function.

        Returns:
            The modified emitter state.
        """
        return emitter_state
