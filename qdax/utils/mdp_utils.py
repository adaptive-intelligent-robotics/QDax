from functools import partial
from typing import Any, Callable, Tuple

import brax
import flax
import jax
import jax.numpy as jnp
from brax.envs import State as EnvState

from qdax.types import (
    Action,
    Descriptor,
    Done,
    Fitness,
    Genotype,
    Observation,
    Params,
    Reward,
    RNGKey,
    StateDescriptor,
)


@flax.struct.dataclass
class Transition:
    """Stores data corresponding to a transition collected by a classic RL algorithm."""

    obs: Observation
    next_obs: Observation
    rewards: Reward
    dones: Done
    truncations: jnp.ndarray  # Indicates if an episode has reached max time step
    actions: Action

    @property
    def observation_dim(self) -> int:
        """
        Returns:
            the dimension of the observation
        """
        return self.obs.shape[-1]

    @property
    def action_dim(self) -> int:
        """
        Returns:
            the dimension of the action
        """
        return self.actions.shape[-1]

    @property
    def flatten_dim(self) -> int:
        """
        Returns:
            the dimension of the transition once flattened.

        """
        flatten_dim = 2 * self.observation_dim + self.action_dim + 3
        return flatten_dim

    def flatten(self) -> jnp.ndarray:
        """
        Returns:
            a jnp.ndarray that corresponds to the flattened transition.
        """
        flatten_transition = jnp.concatenate(
            [
                self.obs,
                self.next_obs,
                jnp.expand_dims(self.rewards, axis=-1),
                jnp.expand_dims(self.dones, axis=-1),
                jnp.expand_dims(self.truncations, axis=-1),
                self.actions,
            ],
            axis=-1,
        )
        return flatten_transition

    @classmethod
    def from_flatten(
        cls,
        flattened_transition: jnp.ndarray,
        transition: "Transition",
    ) -> "Transition":
        """
        Creates a transition from a flattened transition in a jnp.ndarray.

        Args:
            flattened_transition: flattened transition in a jnp.ndarray of shape
                (batch_size, flatten_dim)
            transition: a transition object (might be a dummy one) to
                get the dimensions right

        Returns:
            a Transition object
        """
        obs_dim = transition.observation_dim
        action_dim = transition.action_dim
        obs = flattened_transition[:, :obs_dim]
        next_obs = flattened_transition[:, obs_dim : (2 * obs_dim)]
        rewards = jnp.ravel(flattened_transition[:, (2 * obs_dim) : (2 * obs_dim + 1)])
        dones = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 1) : (2 * obs_dim + 2)]
        )
        truncations = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 2) : (2 * obs_dim + 3)]
        )
        actions = flattened_transition[
            :, (2 * obs_dim + 3) : (2 * obs_dim + 3 + action_dim)
        ]
        return cls(
            obs=obs,
            next_obs=next_obs,
            rewards=rewards,
            dones=dones,
            truncations=truncations,
            actions=actions,
        )

    @classmethod
    def init_dummy(cls, observation_dim: int, action_dim: int) -> "Transition":
        """
        Initialize a dummy transition that then can be passed to constructors to get
        all shapes right.

        Args:
            observation_dim: observation dimension
            action_dim: action dimension

        Returns:
            a dummy transition
        """
        dummy_transition = Transition(
            obs=jnp.zeros(shape=(1, observation_dim)),
            next_obs=jnp.zeros(shape=(1, observation_dim)),
            rewards=jnp.zeros(shape=(1,)),
            dones=jnp.zeros(shape=(1,)),
            truncations=jnp.zeros(shape=(1,)),
            actions=jnp.zeros(shape=(1, action_dim)),
        )
        return dummy_transition


@flax.struct.dataclass
class QDTransition(Transition):
    """Stores data corresponding to a transition collected by a QD algorithm."""

    state_desc: StateDescriptor
    next_state_desc: StateDescriptor

    @property
    def state_descriptor_dim(self) -> int:
        """
        Returns:
            the dimension of the state descriptors.

        """
        return self.state_desc.shape[-1]

    @property
    def flatten_dim(self) -> int:
        """
        Returns:
            the dimension of the transition once flattened.

        """
        flatten_dim = (
            2 * self.observation_dim
            + self.action_dim
            + 3
            + 2 * self.state_descriptor_dim
        )
        return flatten_dim

    def flatten(self) -> jnp.ndarray:
        """
        Returns:
            a jnp.ndarray that corresponds to the flattened transition.
        """
        flatten_transition = jnp.concatenate(
            [
                self.obs,
                self.next_obs,
                jnp.expand_dims(self.rewards, axis=-1),
                jnp.expand_dims(self.dones, axis=-1),
                jnp.expand_dims(self.truncations, axis=-1),
                self.actions,
                self.state_desc,
                self.next_state_desc,
            ],
            axis=-1,
        )
        return flatten_transition

    @classmethod
    def from_flatten(
        cls,
        flattened_transition: jnp.ndarray,
        transition: "QDTransition",
    ) -> "QDTransition":
        """
        Creates a transition from a flattened transition in a jnp.ndarray.

        Args:
            flattened_transition: flattened transition in a jnp.ndarray of shape
                (batch_size, flatten_dim)
            transition: a transition object (might be a dummy one) to
                get the dimensions right

        Returns:
            a Transition object
        """
        obs_dim = transition.observation_dim
        action_dim = transition.action_dim
        desc_dim = transition.state_descriptor_dim

        obs = flattened_transition[:, :obs_dim]
        next_obs = flattened_transition[:, obs_dim : (2 * obs_dim)]
        rewards = jnp.ravel(flattened_transition[:, (2 * obs_dim) : (2 * obs_dim + 1)])
        dones = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 1) : (2 * obs_dim + 2)]
        )
        truncations = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 2) : (2 * obs_dim + 3)]
        )
        actions = flattened_transition[
            :, (2 * obs_dim + 3) : (2 * obs_dim + 3 + action_dim)
        ]
        state_desc = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim) : (2 * obs_dim + 3 + action_dim + desc_dim),
        ]
        next_state_desc = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim + desc_dim) : (
                2 * obs_dim + 3 + action_dim + 2 * desc_dim
            ),
        ]
        return cls(
            obs=obs,
            next_obs=next_obs,
            rewards=rewards,
            dones=dones,
            truncations=truncations,
            actions=actions,
            state_desc=state_desc,
            next_state_desc=next_state_desc,
        )

    @classmethod
    def init_dummy(
        cls, observation_dim: int, action_dim: int, descriptor_dim: int
    ) -> "QDTransition":
        """
        Initialize a dummy transition that then can be passed to constructors to get
        all shapes right.

        Args:
            observation_dim: observation dimension
            action_dim: action dimension

        Returns:
            a dummy transition
        """
        dummy_transition = QDTransition(
            obs=jnp.zeros(shape=(1, observation_dim)),
            next_obs=jnp.zeros(shape=(1, observation_dim)),
            rewards=jnp.zeros(shape=(1,)),
            dones=jnp.zeros(shape=(1,)),
            truncations=jnp.zeros(shape=(1,)),
            actions=jnp.zeros(shape=(1, action_dim)),
            state_desc=jnp.zeros(shape=(1, descriptor_dim)),
            next_state_desc=jnp.zeros(shape=(1, descriptor_dim)),
        )
        return dummy_transition


@partial(jax.jit, static_argnames=("play_step_fn", "episode_length"))
def generate_unroll(
    init_state: EnvState,
    policy_params: Params,
    key: RNGKey,
    episode_length: int,
    play_step_fn: Callable[
        [EnvState, Params, RNGKey],
        Tuple[
            EnvState,
            Params,
            RNGKey,
            Transition,
        ],
    ],
) -> Tuple[EnvState, Transition]:
    """
    Generates an episode according to the agent's policy, returns the final state
    of the episode and the transitions of the episode.
    """

    def _scan_play_step_fn(
        carry: Tuple[EnvState, Params, RNGKey], unused_arg: Any
    ) -> Tuple[Tuple[EnvState, Params, RNGKey], Transition]:
        env_state, policy_params, random_key, transitions = play_step_fn(*carry)
        return (env_state, policy_params, random_key), transitions

    (state, _, _), transitions = jax.lax.scan(
        _scan_play_step_fn,
        (init_state, policy_params, key),
        (),
        length=episode_length,
    )
    return state, transitions


@partial(
    jax.jit,
    static_argnames=(
        "episode_length",
        "play_step_fn",
        "behavior_descriptor_extractor",
    ),
)
def scoring_function(
    policies_params: Genotype,
    init_states: brax.envs.State,
    episode_length: int,
    random_key: RNGKey,
    play_step_fn: Callable[
        [EnvState, Params, RNGKey, brax.envs.Env],
        Tuple[EnvState, Params, RNGKey, Transition],
    ],
    behavior_descriptor_extractor: Callable[[Transition, jnp.ndarray], Descriptor],
) -> Tuple[Fitness, Descriptor]:
    """
    Evaluate policies contained in flatten_variables in parallel

    This rollout is only determinist when all the init states are the same.
    If the init states are fixed but different, as a policy is not necessarly
    evaluated with the same environment everytime, this won't be determinist.

    When the init states, this is not purely stochastic. This choice was made
    for performance reason, as the reset function of brax envs is quite time
    consuming. If pure stochasticity is needed for a use case, please open
    an issue.
    """

    # Perform rollouts with each policy
    unroll_fn = partial(
        generate_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        key=random_key,
    )

    _final_state, data = jax.vmap(unroll_fn)(init_states, policies_params)

    # create a mask to extract data properly
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=1), 0, 1)
    mask = jnp.roll(is_done, 1, axis=1)
    mask = mask.at[:, 0].set(0)

    # Scores - add offset to ensure positive fitness (through positive rewards)
    fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1)
    descriptors = behavior_descriptor_extractor(data, mask)

    return fitnesses, descriptors
