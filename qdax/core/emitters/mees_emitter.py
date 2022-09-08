from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


@dataclass
class MEESConfig:
    """Configuration for the MAP-Elites-ES emitter."""

    sample_number: int = 1000  # num of samples for gradient estimate
    sample_sigma: float = 0.02
    sample_mirror: bool = True  # if 1, use mirroring sampling
    sample_rank_norm: bool = True  # if 1, use normalisation
    num_optimizer_steps: int = 10  # frequency of archive-sampling
    adam_optimizer: bool = True  # if 1 use ADAM, if 0 use SGD
    learning_rate: float = 0.01
    learning_rate_decay: float = 1.0  # only applied when no ADAM
    l2_coefficient: float = 0.0  # coefficient for regularisation
    novelty_nearest_neighbors: int = 10
    use_explore: bool = True  # if 0, use only fitness gradient


class MEESEmitterState(EmitterState):
    """Emitter State for the MAP-Elites-ES emitter."""

    learning_rate: float
    initial_optimizer_state: optax.OptState  # only used for ADAM
    optimizer_state: optax.OptState  # only used for ADAM
    gradient_offspring: Genotype  # Offspring generated through gradient
    generation_count: int  # Generation counter
    novelty_archive: Descriptor  # Used to compute novelty for explore
    last_updated_genotypes: Genotype  # Used to sample parents
    last_updated_fitnesses: Fitness
    last_updated_position: jnp.array
    random_key: RNGKey


class MEESEmitter(Emitter):
    """
    Emitter reproducing the MAP-Elites-ES exploit-explore and MAP-Elites-ES exploit
    algorithms from "Scaling MAP-Elites to Deep Neuroevolution" by Colas et al:
    https://dl.acm.org/doi/pdf/10.1145/3377930.3390217

    One can choose between the two algorithms by setting the use_explore param:
        ME-ES exploit-explore: alternates between num_optimizer_steps of fitness
            gradients and num_optimizer_steps of novelty gradients.
        ME-ES eploit: only uses fitness gradient and no novelty gradients, resample
            parent from the archive every num_optimizer_steps steps.
    """

    def __init__(
        self,
        config: MEESConfig,
        total_generations: int,
        scoring_fn: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        num_descriptors: int,
    ) -> None:
        """Initialise the MAP-Elites-ES emitter.

        WARNING: total_generations is required to build the novelty archive.

        Args:
            config
            total_generations: total number of generations for which the
                emitter will run, allow to initialise the novelty archive.
            scoring_fn: used to evaluate the samples for the gradient estimate.
            num_descriptors: dimension of the descriptors, used to initialise
                the empty novelty archive.

        Returns: /
        """
        self._config = config
        self._scoring_fn = scoring_fn
        self._total_generations = total_generations
        self._num_descriptors = num_descriptors

        # Initialise optimizer
        if self._config.adam_optimizer:
            self._optimizer = optax.adam(learning_rate=config.learning_rate)
        else:
            self._optimizer = None

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[MEESEmitterState, RNGKey]:
        """Initializes the emitter state.

        Args:
            init_genotypes: The initial population.
            random_key: A random key.

        Returns:
            The initial state of the MEESEmitter, a new random key.
        """
        assert (
            jax.tree_leaves(init_genotypes)[0].shape[0] == 1
        ), "!!!ERROR!!! MAP-Elites-ES generates one individual per generation, i.e. batch_size = 1."

        learning_rate = self._config.learning_rate

        # Initialise optimizer
        if self._config.adam_optimizer:
            initial_optimizer_state = self._optimizer.init(init_genotypes)
        else:
            initial_optimizer_state = None

        # Create empty Novelty archive
        if self._config.use_explore:
            novelty_archive = jnp.zeros(
                (self._total_generations, self._num_descriptors)
            )
        else:
            novelty_archive = jnp.zeros(1)

        # Create empty updated genotypes and fitness
        last_updated_genotypes = jax.tree_map(
            lambda x: jnp.zeros(shape=(5,) + x.shape[1:]),
            init_genotypes,
        )
        last_updated_fitnesses = -jnp.inf * jnp.ones(shape=5)

        return (
            MEESEmitterState(
                learning_rate=learning_rate,
                initial_optimizer_state=initial_optimizer_state,
                optimizer_state=initial_optimizer_state,
                gradient_offspring=init_genotypes,
                generation_count=0,
                novelty_archive=novelty_archive,
                last_updated_genotypes=last_updated_genotypes,
                last_updated_fitnesses=last_updated_fitnesses,
                last_updated_position=0,
                random_key=random_key,
            ),
            random_key,
        )

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: MEESEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Return the offspring generated through gradient update.

        Params:
            repertoire: the MAP-Elites repertoire to sample from
            emitter_state
            random_key: a jax PRNG random key

        Returns:
            a new gradient offspring
            a new jax PRNG key
        """

        return emitter_state.gradient_offspring, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def _novelty(
        self,
        generation_count: int,
        descriptors: Descriptor,
        novelty_archive: Descriptor,
    ) -> jnp.ndarray:
        """Compute the novelty of descriptors using a novelty archive.

        Args:
            generation_count: used to remove empty novelty archive slots
            descriptors: array of descriptors of the individuals
            novelty_archive: current novelty archive

        Returns: the novelty score of all individuals
        """

        # Compute all distances with novelty_archive content
        num_indivs = descriptors.shape[0]
        distances = jnp.repeat(
            jnp.expand_dims(descriptors, axis=1),
            self._total_generations,
            axis=1,
            total_repeat_length=self._total_generations,
        )
        distances = distances - jnp.repeat(
            jnp.expand_dims(novelty_archive, axis=0),
            num_indivs,
            axis=0,
            total_repeat_length=num_indivs,
        )
        distances = jnp.sqrt(jnp.sum(jnp.square(distances), axis=2))

        # Filter distance with empty slot of novelty archive
        all_indices = jnp.repeat(
            jnp.expand_dims(
                jnp.arange(0, self._total_generations, step=1),
                axis=0,
            ),
            num_indivs,
            axis=0,
            total_repeat_length=num_indivs,
        )
        distances = jnp.where(all_indices >= generation_count, jnp.inf, distances)

        # Find k neirest neighbours
        _, indices = jax.lax.top_k(-distances, self._config.novelty_nearest_neighbors)

        # Compute novelty as average distance with k neirest neirghbours
        distances = jnp.where(distances == jnp.inf, jnp.nan, distances)
        novelty = jnp.nanmean(jnp.take_along_axis(distances, indices, axis=1), axis=1)
        return novelty

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def _sample_exploit(
        self,
        last_updated_genotypes: Genotype,
        last_updated_fitnesses: Fitness,
        repertoire: MapElitesRepertoire,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Sample half of the time uniformly from the two highest fitness cells
        of the repertoire and half of the time uniformly from the two highest
        fitness cells among the last five updated cells.

        Args:
            last_updated_genotypes: 5 last updated genotypes
            last_updated_fitnesses: corresponding fitnesses
            repertoire: the current repertoire
            random_key: a jax PRNG random key

        Returns:
            samples: a genotype sampled in the repertoire
            random_key: an updated jax PRNG random key
        """

        def _sample(
            genotypes: Genotype,
            fitnesses: Fitness,
            random_key: RNGKey,
        ) -> Tuple[Genotype, RNGKey]:
            """Sample uniformly from the 2 highest fitness cells."""

            max_fitnesses, _ = jax.lax.top_k(fitnesses, 2)
            min_fitness = jnp.nanmin(jnp.where(max_fitnesses > -jnp.inf, max_fitnesses, jnp.inf))
            genotypes_empty = fitnesses < min_fitness
            p = (1.0 - genotypes_empty) / jnp.sum(1.0 - genotypes_empty)
            random_key, subkey = jax.random.split(random_key)
            samples = jax.tree_map(
                lambda x: jax.random.choice(subkey, x, shape=(1,), p=p),
                genotypes,
            )
            return samples, random_key

        random_key, subkey = jax.random.split(random_key)

        # Sample p uniformly
        p = jax.random.uniform(subkey)

        # Depending on the value of p, use one of the two sampling options
        samples, random_key = jax.lax.cond(
            p < 0.5,
            lambda random_key: _sample(
                repertoire.genotypes, repertoire.fitnesses, random_key
            ),
            lambda random_key: _sample(
                last_updated_genotypes, last_updated_fitnesses, random_key
            ),
            (random_key),
        )

        return samples, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def _sample_explore(
        self,
        generation_count: int,
        novelty_archive: Descriptor,
        repertoire: MapElitesRepertoire,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Sample uniformly from the 5 most novel genotypes.

        Args:
            emitter_state: current emitter state
            repertoire: the current genotypes repertoire
            random_key: a jax PRNG random key

        Returns:
            samples: a genotype sampled in the repertoire
            random_key: an updated jax PRNG random key
        """

        # Compute the novelty of all indivs in the archive
        novelties = self._novelty(
            generation_count + 1,
            repertoire.descriptors,
            novelty_archive,
        )
        novelties = jnp.where(repertoire.fitnesses > -jnp.inf, novelties, -jnp.inf)

        # Sample uniformaly for the 5 most novel cells
        max_novelties, _ = jax.lax.top_k(novelties, 5)
        min_novelty = jnp.nanmin(jnp.where(max_novelties > -jnp.inf, max_novelties, jnp.inf))
        repertoire_empty = novelties < min_novelty
        p = (1.0 - repertoire_empty) / jnp.sum(1.0 - repertoire_empty)
        random_key, subkey = jax.random.split(random_key)
        samples = jax.tree_map(
            lambda x: jax.random.choice(subkey, x, shape=(1,), p=p),
            repertoire.genotypes,
        )

        return samples, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: MEESEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> MEESEmitterState:
        """Generate the gradient offspring for the next emitter call. Also
        update the novelty archive and generation count from current call.

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

        assert (
            jax.tree_leaves(genotypes)[0].shape[0] == 1
        ), "!!!ERROR!!! MAP-Elites-ES generates one individual per generation, i.e. batch_size = 1."

        ############################
        # Updating novelty archive #

        generation_count = emitter_state.generation_count
        novelty_archive = emitter_state.novelty_archive
        if self._config.use_explore:
            novelty_archive = jax.lax.dynamic_update_slice_in_dim(
                novelty_archive,
                descriptors,
                generation_count,
                axis=0,
            )

        ################################
        # Updating last updated indivs #

        # Get indice of last genotype
        indice = get_cells_indices(descriptors, repertoire.centroids)

        # Check if it has been added to the grid
        added_genotype = jnp.all(
            jnp.asarray(
                jax.tree_util.tree_leaves(
                    jax.tree_map(
                        lambda new_gen, rep_gen: jnp.all(
                            jnp.equal(
                                jnp.ravel(new_gen), jnp.ravel(rep_gen.at[indice].get())
                            ),
                            axis=0,
                        ),
                        genotypes,
                        repertoire.genotypes,
                    ),
                )
            ),
            axis=0,
        )

        # Update last_added buffers
        last_updated_position = jnp.where(
            added_genotype, emitter_state.last_updated_position, 6
        )
        last_updated_fitnesses = emitter_state.last_updated_fitnesses
        last_updated_fitnesses = last_updated_fitnesses.at[last_updated_position].set(
            fitnesses[0]
        )
        last_updated_genotypes = jax.tree_map(
            lambda last_gen, gen: last_gen.at[
                jnp.expand_dims(last_updated_position, axis=0)
            ].set(gen),
            emitter_state.last_updated_genotypes,
            genotypes,
        )
        last_updated_position = (
            emitter_state.last_updated_position + added_genotype
        ) % 5

        ########################
        # Sampling new parent #

        random_key = emitter_state.random_key
        generation_count = emitter_state.generation_count

        # Select between new sampled parent and previous parent
        parent, random_key = jax.lax.cond(
            generation_count % self._config.num_optimizer_steps == 0,
            lambda random_key: jax.lax.cond(
                (generation_count // self._config.num_optimizer_steps) % 2 == 0,
                lambda random_key: self._sample_explore(
                    generation_count, novelty_archive, repertoire, random_key
                ),
                lambda random_key: self._sample_exploit(
                    last_updated_genotypes,
                    last_updated_fitnesses,
                    repertoire,
                    random_key,
                ),
                (random_key),
            ),
            lambda random_key: (emitter_state.gradient_offspring, random_key),
            (random_key),
        )

        # Select between new optimizer state and previous optimizer state
        optimizer_state = jax.lax.cond(
            generation_count % self._config.num_optimizer_steps == 0,
            lambda unused: emitter_state.initial_optimizer_state,
            lambda unused: emitter_state.optimizer_state,
            (),
        )
        random_key, subkey = jax.random.split(random_key)

        ##########################################
        # Creating samples for gradient estimate #

        # Sampling mirror noise
        total_sample_number = self._config.sample_number
        if self._config.sample_mirror:
            sample_number = total_sample_number // 2

            half_sample_noise = jax.tree_map(
                lambda x: jax.random.normal(
                    key=subkey,
                    shape=jnp.repeat(x, sample_number, axis=0).shape,
                ),
                parent,
            )
            # Splitting noise to apply it in mirror to samples
            sample_noise = jax.tree_map(
                lambda x: jnp.concatenate(
                    [jnp.expand_dims(x, axis=1), jnp.expand_dims(-x, axis=1)], axis=1
                ).reshape(jnp.repeat(x, 2, axis=0).shape),
                half_sample_noise,
            )
            # Noise used for gradient computation
            gradient_noise = half_sample_noise

        # Sampling non-mirror noise
        else:
            sample_number = total_sample_number

            sample_noise = jax.tree_map(
                lambda x: jax.random.normal(
                    key=subkey,
                    shape=jnp.repeat(x, sample_number, axis=0).shape,
                ),
                parent,
            )
            # Noise used for gradient computation
            gradient_noise = sample_noise

        # Expanding dimension to number of samples
        samples = jax.tree_map(
            lambda x: jnp.repeat(x, total_sample_number, axis=0),
            parent,
        )

        # Applying noise to each sample
        samples = jax.tree_map(
            lambda mean, noise: mean + self._config.sample_sigma * noise,
            samples,
            sample_noise,
        )

        ######################
        # Evaluating samples #

        fitnesses, descriptors, extra_scores, random_key = self._scoring_fn(
            samples, random_key
        )

        #######################
        # Computing gradients #

        # Choosing the score to use for rank
        scores = jax.lax.cond(
            (generation_count // self._config.num_optimizer_steps) % 2 == 0,
            lambda scoring: self._novelty(
                generation_count + 1, scoring[1], emitter_state.novelty_archive
            ),
            lambda scoring: scoring[0],
            (fitnesses, descriptors),
        )

        # Computing rank with normalisation
        if self._config.sample_rank_norm:

            # Ranking objective
            ranking_indices = jnp.argsort(scores, axis=0)
            ranks = jnp.argsort(ranking_indices, axis=0)

            # Normalising ranks to [-0.5, 0.5]
            ranks = (ranks / (total_sample_number - 1)) - 0.5

        # Computing rank without normalisation
        else:
            ranks = scores

        # Reshaping rank to match shape of genotype_noise
        if self._config.sample_mirror:
            ranks = jnp.reshape(ranks, (sample_number, 2))
            ranks = jnp.apply_along_axis(lambda rank: rank[0] - rank[1], 1, ranks)

        ranks = jax.tree_map(
            lambda x: jnp.reshape(
                jnp.repeat(ranks.ravel(), x[0].ravel().shape[0], axis=0), x.shape
            ),
            gradient_noise,
        )

        # Computing the gradients
        gradient = jax.tree_map(
            lambda noise, rank: jnp.multiply(noise, rank),
            gradient_noise,
            ranks,
        )
        gradient = jax.tree_map(
            lambda x: jnp.reshape(x, (sample_number, -1)),
            gradient,
        )
        gradient = jax.tree_map(
            lambda g, p: jnp.reshape(
                -jnp.sum(g, axis=0) / (total_sample_number * self._config.sample_sigma),
                p.shape,
            ),
            gradient,
            parent,
        )

        # Adding regularisation
        gradient = jax.tree_map(
            lambda g, p: g + self._config.l2_coefficient * p,
            gradient,
            parent,
        )

        ######################
        # Applying gradients #

        if self._config.adam_optimizer:
            (offspring_update, optimizer_state) = self._optimizer.update(
                gradient, optimizer_state
            )
            offspring = optax.apply_updates(parent, offspring_update)
            learning_rate = self._config.learning_rate
        else:
            offspring = jax.tree_map(
                lambda p, g: p - emitter_state.learning_rate * g,
                parent,
                gradient,
            )
            learning_rate = (
                emitter_state.learning_rate * self._config.learning_rate_decay
            )

        # Increase generation counter
        generation_count += 1

        return emitter_state.replace(  # type: ignore
            learning_rate=learning_rate,
            optimizer_state=optimizer_state,
            gradient_offspring=offspring,
            generation_count=generation_count,
            novelty_archive=novelty_archive,
            last_updated_genotypes=last_updated_genotypes,
            last_updated_fitnesses=last_updated_fitnesses,
            last_updated_position=last_updated_position,
            random_key=random_key,
        )
