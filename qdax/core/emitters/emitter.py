from abc import ABC, abstractmethod
from functools import partial
from typing import Optional, Tuple

import jax
from flax.struct import PyTreeNode

from qdax.core.containers.repertoire import Repertoire
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class EmitterState(PyTreeNode):
    """The state of an emitter. Emitters are used to suggest offspring
    when evolving a population of genotypes. To emit new genotypes, some
    emitters need to have a state, that carries useful informations, like
    running means, distribution parameters, critics, replay buffers etc...

    The object emitter state is used to store them and is updated along
    the process.

    Args:
        PyTreeNode: EmitterState base class inherits from PyTreeNode object
            from flax.struct package. It help registering objects as Pytree
            nodes automatically, and as the same benefits as classic Python
            @dataclass decorator.
    """

    pass


class Emitter(ABC):
    def init(
        self, init_genotypes: Optional[Genotype], random_key: RNGKey
    ) -> Tuple[Optional[EmitterState], RNGKey]:
        """Initialises the state of the emitter. Some emitters do
        not need a state, in which case, the value None can be
        outputted.

        Args:
            init_genotypes: The genotypes of the initial population.
            random_key: a random key to handle stochastic operations.

        Returns:
            The initial emitter state and a random key.
        """
        return None, random_key

    @abstractmethod
    def emit(
        self,
        repertoire: Optional[Repertoire],
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Function used to emit a population of offspring by any possible
        mean. New population can be sampled from a distribution or obtained
        through mutations of individuals sampled from the repertoire.


        Args:
            repertoire: a repertoire of genotypes.
            emitter_state: the state of the emitter.
            random_key: a random key to handle random operations.

        Returns:
            A batch of offspring, a new random key.
        """
        pass

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: Optional[EmitterState],
        repertoire: Optional[Repertoire] = None,
        genotypes: Optional[Genotype] = None,
        fitnesses: Optional[Fitness] = None,
        descriptors: Optional[Descriptor] = None,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Optional[EmitterState]:
        """This function gives an opportunity to update the emitter state
        after the genotypes have been scored.

        As a matter of fact, many emitter states needs informations from
        the evaluations of the genotypes in order to be updated, for instance:
        - CMA emitter: to update the rank of the covariance matrix
        - PGA emitter: to fill the replay buffer and update the critic/greedy
            couple.

        This function does not need to be overridden. By default, it output
        the same emitter state.

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
        return emitter_state

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        pass

    @property
    def use_all_data(self) -> bool:
        """Whether to use all data or not when used along other emitters.

        Used when an emitter is used in a multi emitter setting.

        Some emitter only the information from the genotypes they emitted when
        they update their state (for instance, the CMA emitters); but other use data
        from genotypes emitted by others (for instance, QualityPGEmitter and
        DiversityPGEmitter). The meta emitters like MultiEmitter need to know which
        data to give the sub emitter when udapting them. This property is used at
        this moment.

        Default behavior is to used only the data related to what was emitted.

        Returns:
            Whether to pass only the genotypes (and their evaluations) that the emitter
            emitted when updating it or all the genotypes emitted by all the emitters.
        """
        return False
