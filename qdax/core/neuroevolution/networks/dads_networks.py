from typing import Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from haiku.initializers import Initializer, VarianceScaling

from qdax.types import Action, Observation, Skill, StateDescriptor


class GaussianMixture(hk.Module):
    """Module that outputs a Gaussian Mixture Distribution."""

    def __init__(
        self,
        num_dimensions: int,
        num_components: int,
        reinterpreted_batch_ndims: Optional[int] = None,
        identity_covariance: bool = True,
        initializer: Optional[Initializer] = None,
        name: str = "GaussianMixture",
    ):
        """Module that outputs a Gaussian Mixture Distribution
        with identity covariance matrix."""

        super().__init__(name=name)
        if initializer is None:
            initializer = VarianceScaling(1.0, "fan_in", "uniform")
        self._num_dimensions = num_dimensions
        self._num_components = num_components
        self._reinterpreted_batch_ndims = reinterpreted_batch_ndims
        self._identity_covariance = identity_covariance
        self.initializer = initializer
        logits_size = self._num_components

        self.logit_layer = hk.Linear(logits_size, w_init=self.initializer)

        # Create two layers that outputs a location and a scale, respectively, for
        # each dimension and each component.
        self.loc_layer = hk.Linear(
            self._num_dimensions * self._num_components, w_init=self.initializer
        )
        if not self._identity_covariance:
            self.scale_layer = hk.Linear(
                self._num_dimensions * self._num_components, w_init=self.initializer
            )

    def __call__(self, inputs: jnp.ndarray) -> tfp.distributions.Distribution:
        # Compute logits, locs, and scales if necessary.
        logits = self.logit_layer(inputs)
        locs = self.loc_layer(inputs)

        shape = [-1, self._num_components, self._num_dimensions]  # [B, D, C]

        # Reshape the mixture's location and scale parameters appropriately.
        locs = locs.reshape(shape)
        if not self._identity_covariance:

            scales = self.scale_layer(inputs)
            scales = scales.reshape(shape)
        else:
            scales = jnp.ones_like(locs)

        # Create the mixture distribution
        components = tfp.distributions.MultivariateNormalDiag(
            loc=locs, scale_diag=scales
        )
        mixture = tfp.distributions.Categorical(logits=logits)
        distribution = tfp.distributions.MixtureSameFamily(
            mixture_distribution=mixture, components_distribution=components
        )

        return distribution


class DynamicsNetwork(hk.Module):
    """Dynamics network (used in DADS)."""

    def __init__(
        self,
        hidden_layer_sizes: tuple,
        output_size: int,
        omit_input_dynamics_dim: int = 2,
        name: Optional[str] = None,
        identity_covariance: bool = True,
        initializer: Optional[Initializer] = None,
    ):
        super().__init__(name=name)
        if initializer is None:
            initializer = VarianceScaling(1.0, "fan_in", "uniform")

        self.distribution = GaussianMixture(
            output_size,
            num_components=4,
            reinterpreted_batch_ndims=None,
            identity_covariance=identity_covariance,
            initializer=initializer,
        )
        self.network = hk.Sequential(
            [
                hk.nets.MLP(
                    list(hidden_layer_sizes),
                    w_init=initializer,
                    activation=jax.nn.relu,
                    activate_final=True,
                ),
            ]
        )
        self._omit_input_dynamics_dim = omit_input_dynamics_dim

    def __call__(
        self, obs: StateDescriptor, skill: Skill, target: StateDescriptor
    ) -> jnp.ndarray:
        """Normalizes the observation, predicts a distribution probability conditioned
        on (obs,skill) and returns the log_prob of the target.
        """

        obs = obs[:, self._omit_input_dynamics_dim :]
        obs = jnp.concatenate((obs, skill), axis=1)
        out = self.network(obs)
        dist = self.distribution(out)
        return dist.log_prob(target)


def make_dads_networks(
    action_size: int,
    descriptor_size: int,
    hidden_layer_sizes: Tuple[int, ...] = (256, 256),
    omit_input_dynamics_dim: int = 2,
    identity_covariance: bool = True,
    dynamics_initializer: Optional[Initializer] = None,
) -> Tuple[hk.Transformed, hk.Transformed, hk.Transformed]:
    """Creates networks used in DADS.

    Args:
        action_size: the size of the environment's action space
        descriptor_size: the size of the environment's descriptor space (i.e. the
            dimension of the dynamics network's input)
        hidden_layer_sizes: the number of neurons for hidden layers.
            Defaults to (256, 256).
        omit_input_dynamics_dim: how many descriptors we omit when creating the input
            of the dynamics networks. Defaults to 2.
        identity_covariance: whether to fix the covariance matrix of the Gaussian models
            to identity. Defaults to True.
        dynamics_initializer: the initializer of the dynamics layers. Defaults to None.

    Returns:
        the policy network
        the critic network
        the dynamics network
    """

    def _actor_fn(obs: Observation) -> jnp.ndarray:
        network = hk.Sequential(
            [
                hk.nets.MLP(
                    list(hidden_layer_sizes) + [2 * action_size],
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                ),
            ]
        )
        return network(obs)

    def _critic_fn(obs: Observation, action: Action) -> jnp.ndarray:
        network1 = hk.Sequential(
            [
                hk.nets.MLP(
                    list(hidden_layer_sizes) + [1],
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                ),
            ]
        )
        network2 = hk.Sequential(
            [
                hk.nets.MLP(
                    list(hidden_layer_sizes) + [1],
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                ),
            ]
        )
        input_ = jnp.concatenate([obs, action], axis=-1)
        value1 = network1(input_)
        value2 = network2(input_)
        return jnp.concatenate([value1, value2], axis=-1)

    def _dynamics_fn(
        obs: StateDescriptor, skill: Skill, target: StateDescriptor
    ) -> jnp.ndarray:
        dynamics_network = DynamicsNetwork(
            hidden_layer_sizes,
            descriptor_size,
            omit_input_dynamics_dim=omit_input_dynamics_dim,
            identity_covariance=identity_covariance,
            initializer=dynamics_initializer,
        )
        return dynamics_network(obs, skill, target)

    policy = hk.without_apply_rng(hk.transform(_actor_fn))
    critic = hk.without_apply_rng(hk.transform(_critic_fn))
    dynamics = hk.without_apply_rng(hk.transform(_dynamics_fn))

    return policy, critic, dynamics
