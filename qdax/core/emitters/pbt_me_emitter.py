from functools import partial
from typing import Callable, Optional, Tuple, Union

import jax
from brax.envs import State as EnvState
from flax.struct import PyTreeNode
from jax import numpy as jnp

from qdax.baselines.pbt import PBTTrainingState
from qdax.baselines.sac_pbt import PBTSAC
from qdax.baselines.td3_pbt import PBTTD3
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.neuroevolution.buffers.buffer import ReplayBuffer, Transition
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey


class PBTEmitterState(EmitterState):
    """
    PBT emitter state contains the replay buffers that will be used by the population as
    well as the population agents training states and their starting environment state.
    """

    replay_buffers: ReplayBuffer
    env_states: EnvState
    training_states: PBTTrainingState
    random_key: RNGKey


class PBTEmitterConfig(PyTreeNode):
    """
    Config for the PBT-ME emitter. This mainly corresponds to the hyperparameters
    of the PBT-ME algorithm.
    """

    buffer_size: int
    num_training_iterations: int
    env_batch_size: int
    grad_updates_per_step: int
    pg_population_size_per_device: int
    ga_population_size_per_device: int
    num_devices: int

    fraction_best_to_replace_from: float
    fraction_to_replace_from_best: float
    fraction_to_replace_from_samples: float
    # this fraction is used only for transfer between devices
    fraction_sort_exchange: float


class PBTEmitter(Emitter):
    """
    A policy gradient emitter used to implement the Policy Gradient Assisted MAP-Elites
    (PGA-Map-Elites) algorithm.
    """

    def __init__(
        self,
        pbt_agent: Union[PBTSAC, PBTTD3],
        config: PBTEmitterConfig,
        env: QDEnv,
        variation_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
    ) -> None:

        # Parameters internalization
        self._env = env
        self._variation_fn = variation_fn
        self._config = config
        self._agent = pbt_agent
        self._train_fn = self._agent.get_train_fn(
            env=env,
            num_iterations=config.num_training_iterations,
            env_batch_size=config.env_batch_size,
            grad_updates_per_step=config.grad_updates_per_step,
        )

        # Compute numbers from fractions
        pg_population_size = config.pg_population_size_per_device * config.num_devices
        self._num_best_to_replace_from = int(
            pg_population_size * config.fraction_best_to_replace_from
        )
        self._num_to_replace_from_best = int(
            pg_population_size * config.fraction_to_replace_from_best
        )
        self._num_to_replace_from_samples = int(
            pg_population_size * config.fraction_to_replace_from_samples
        )
        self._num_to_exchange = int(
            config.pg_population_size_per_device * config.fraction_sort_exchange
        )

    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[PBTEmitterState, RNGKey]:
        """Initializes the emitter state.

        Args:
            init_genotypes: The initial population.
            random_key: A random key.

        Returns:
            The initial state of the PGAMEEmitter, a new random key.
        """

        observation_size = self._env.observation_size
        action_size = self._env.action_size

        # Initialise replay buffers
        init_dummy_transition = partial(
            Transition.init_dummy,
            observation_dim=observation_size,
            action_dim=action_size,
        )
        init_dummy_transition = jax.vmap(
            init_dummy_transition, axis_size=self._config.pg_population_size_per_device
        )
        dummy_transitions = init_dummy_transition()

        replay_buffer_init = partial(
            ReplayBuffer.init,
            buffer_size=self._config.buffer_size,
        )
        replay_buffer_init = jax.vmap(replay_buffer_init)
        replay_buffers = replay_buffer_init(transition=dummy_transitions)

        # Initialise env states
        (random_key, subkey1, subkey2) = jax.random.split(random_key, num=3)
        env_states = jax.jit(self._env.reset)(rng=subkey1)

        reshape_fn = jax.jit(
            lambda tree: jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x,
                    (
                        self._config.pg_population_size_per_device,
                        self._config.env_batch_size,
                    )
                    + x.shape[1:],
                ),
                tree,
            ),
        )
        env_states = reshape_fn(env_states)

        # Create emitter state
        # keep only pg population size training states if more are provided
        init_genotypes = jax.tree_util.tree_map(
            lambda x: x[: self._config.pg_population_size_per_device], init_genotypes
        )
        emitter_state = PBTEmitterState(
            replay_buffers=replay_buffers,
            env_states=env_states,
            training_states=init_genotypes,
            random_key=subkey2,
        )

        return emitter_state, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: PBTEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Do a single PGA-ME iteration: train critics and greedy policy,
        make mutations (evo and pg), score solution, fill replay buffer and insert back
        in the MAP-Elites grid.

        Args:
            repertoire: the current repertoire of genotypes
            emitter_state: the state of the emitter used
            random_key: a random key

        Returns:
            A batch of offspring, the new emitter state and a new key.
        """

        # Mutation PG (the mutation has already been performed during the state update)
        x_mutation_pg = emitter_state.training_states

        # Mutation evo
        if self._config.ga_population_size_per_device > 0:
            mutation_ga_batch_size = self._config.ga_population_size_per_device
            x1, random_key = repertoire.sample(random_key, mutation_ga_batch_size)
            x2, random_key = repertoire.sample(random_key, mutation_ga_batch_size)
            x_mutation_ga, random_key = self._variation_fn(x1, x2, random_key)

            # Gather offspring
            genotypes = jax.tree_util.tree_map(
                lambda x, y: jnp.concatenate([x, y], axis=0),
                x_mutation_ga,
                x_mutation_pg,
            )
        else:
            genotypes = x_mutation_pg

        return genotypes, random_key

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        mutation_pg_batch_size = self._config.pg_population_size_per_device
        mutation_ga_batch_size = self._config.ga_population_size_per_device
        return mutation_pg_batch_size + mutation_ga_batch_size

    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: PBTEmitterState,
        repertoire: Repertoire,
        genotypes: Optional[Genotype],
        fitnesses: Fitness,
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> PBTEmitterState:
        """
        Update the internal emitter state. I.e. update the population replay buffers and
        agents.

        Args:
            emitter_state: current emitter state.
            repertoire: the current genotypes repertoire
            genotypes: unused here - but compulsory in the signature.
            fitnesses: unused here - but compulsory in the signature.
            descriptors: unused here - but compulsory in the signature.
            extra_scores: extra information coming from the scoring function,
                this contains the transitions added to the replay buffer.

        Returns:
            New emitter state where the replay buffer has been filled with
            the new experienced transitions.
        """
        # Look only at the fitness corresponding to emitter state individuals
        fitnesses = fitnesses[self._config.ga_population_size_per_device :]
        fitnesses = jnp.ravel(fitnesses)
        training_states = emitter_state.training_states
        replay_buffers = emitter_state.replay_buffers
        genotypes = (training_states, replay_buffers)

        # Incremental algorithm to gather top best among the population on each device
        # First exchange
        indices_to_share = jnp.arange(self._config.pg_population_size_per_device)
        num_best_local = int(
            self._config.pg_population_size_per_device
            * self._config.fraction_best_to_replace_from
        )
        indices_to_share = indices_to_share[:num_best_local]
        genotypes_to_share, fitnesses_to_share = jax.tree_util.tree_map(
            lambda x: x[indices_to_share], (genotypes, fitnesses)
        )
        gathered_genotypes, gathered_fitnesses = jax.tree_util.tree_map(
            lambda x: jnp.concatenate(jax.lax.all_gather(x, axis_name="p"), axis=0),
            (genotypes_to_share, fitnesses_to_share),
        )

        genotypes_stacked, fitnesses_stacked = gathered_genotypes, gathered_fitnesses
        best_indices_stacked = jnp.argsort(-fitnesses_stacked)
        best_indices_stacked = best_indices_stacked[: self._num_best_to_replace_from]
        best_genotypes_local, best_fitnesses_local = jax.tree_util.tree_map(
            lambda x: x[best_indices_stacked], (genotypes_stacked, fitnesses_stacked)
        )

        # Define loop fn for the other exchanges
        def _loop_fn(i, val):  # type: ignore
            best_genotypes_local, best_fitnesses_local = val
            indices_to_share = jax.lax.dynamic_slice(
                jnp.arange(self._config.pg_population_size_per_device),
                [i * self._num_to_exchange],
                [self._num_to_exchange],
            )
            genotypes_to_share, fitnesses_to_share = jax.tree_util.tree_map(
                lambda x: x[indices_to_share], (genotypes, fitnesses)
            )
            gathered_genotypes, gathered_fitnesses = jax.tree_util.tree_map(
                lambda x: jnp.concatenate(jax.lax.all_gather(x, axis_name="p"), axis=0),
                (genotypes_to_share, fitnesses_to_share),
            )

            genotypes_stacked, fitnesses_stacked = jax.tree_util.tree_map(
                lambda x, y: jnp.concatenate([x, y], axis=0),
                (gathered_genotypes, gathered_fitnesses),
                (best_genotypes_local, best_fitnesses_local),
            )

            best_indices_stacked = jnp.argsort(-fitnesses_stacked)
            best_indices_stacked = best_indices_stacked[
                : self._num_best_to_replace_from
            ]
            best_genotypes_local, best_fitnesses_local = jax.tree_util.tree_map(
                lambda x: x[best_indices_stacked],
                (genotypes_stacked, fitnesses_stacked),
            )
            return (best_genotypes_local, best_fitnesses_local)  # type: ignore

        # Incrementally get the top fraction_best_to_replace_from best individuals
        # on each device
        (best_genotypes_local, best_fitnesses_local) = jax.lax.fori_loop(
            lower=1,
            upper=int(1.0 // self._config.fraction_sort_exchange) + 1,
            body_fun=_loop_fn,
            init_val=(best_genotypes_local, best_fitnesses_local),
        )

        # Gather fitnesses from all devices to rank locally against it
        all_fitnesses = jax.tree_util.tree_map(
            lambda x: jnp.concatenate(jax.lax.all_gather(x, axis_name="p"), axis=0),
            fitnesses,
        )
        all_fitnesses = jnp.ravel(all_fitnesses)
        all_fitnesses = -jnp.sort(-all_fitnesses)
        random_key = emitter_state.random_key
        random_key, sub_key = jax.random.split(random_key)
        best_genotypes = jax.tree_util.tree_map(
            lambda x: jax.random.choice(
                sub_key, x, shape=(len(fitnesses),), replace=True
            ),
            best_genotypes_local,
        )
        best_training_states, best_replay_buffers = best_genotypes

        # Resample hyper-params
        best_training_states = jax.vmap(
            best_training_states.__class__.resample_hyperparams
        )(best_training_states)

        # Replace by individuals from the best
        lower_bound = all_fitnesses[-self._num_to_replace_from_best]
        cond = fitnesses <= lower_bound

        training_states = jax.tree_util.tree_map(
            lambda x, y: jnp.where(
                jnp.expand_dims(
                    cond, axis=tuple([-(i + 1) for i in range(x.ndim - 1)])
                ),
                x,
                y,
            ),
            best_training_states,
            training_states,
        )
        replay_buffers = jax.tree_util.tree_map(
            lambda x, y: jnp.where(
                jnp.expand_dims(
                    cond, axis=tuple([-(i + 1) for i in range(x.ndim - 1)])
                ),
                x,
                y,
            ),
            best_replay_buffers,
            replay_buffers,
        )

        # Replacing with samples from the ME repertoire
        if self._num_to_replace_from_samples > 0:
            me_samples, random_key = repertoire.sample(
                random_key, self._config.pg_population_size_per_device
            )
            # Resample hyper-params
            me_samples = jax.vmap(me_samples.__class__.resample_hyperparams)(me_samples)
            upper_bound = all_fitnesses[
                -self._num_to_replace_from_best - self._num_to_replace_from_samples
            ]
            cond = jnp.logical_and(fitnesses <= upper_bound, fitnesses >= lower_bound)
            training_states = jax.tree_util.tree_map(
                lambda x, y: jnp.where(
                    jnp.expand_dims(
                        cond, axis=tuple([-(i + 1) for i in range(x.ndim - 1)])
                    ),
                    x,
                    y,
                ),
                me_samples,
                training_states,
            )

        # Train the agents
        env_states = emitter_state.env_states
        # Init optimizers state before training the population
        training_states = jax.vmap(training_states.__class__.init_optimizers_states)(
            training_states
        )
        (training_states, env_states, replay_buffers), metrics = self._train_fn(
            training_states, env_states, replay_buffers
        )
        # Empty optimizers states to avoid storing the info in the RAM
        # and having too heavy repertoires
        training_states = jax.vmap(training_states.__class__.empty_optimizers_states)(
            training_states
        )

        # Update emitter state
        emitter_state = emitter_state.replace(
            training_states=training_states,
            replay_buffers=replay_buffers,
            env_states=env_states,
            random_key=random_key,
        )
        return emitter_state  # type: ignore
