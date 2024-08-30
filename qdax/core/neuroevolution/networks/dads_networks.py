from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax.nn import initializers

from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import Action, Observation, Skill, StateDescriptor


class GaussianMixture(nn.Module):
    num_dimensions: int
    num_components: int
    reinterpreted_batch_ndims: Optional[int] = None
    identity_covariance: bool = True
    initializer: Optional[initializers.Initializer] = None

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> tfp.distributions.Distribution:
        if self.initializer is None:
            init = initializers.variance_scaling(1.0, "fan_in", "uniform")
        else:
            init = self.initializer

        logits = nn.Dense(self.num_components, kernel_init=init)(inputs)
        locs = nn.Dense(self.num_dimensions * self.num_components, kernel_init=init)(
            inputs
        )

        shape = [-1, self.num_components, self.num_dimensions]  # [B, D, C]
        locs = locs.reshape(shape)

        if not self.identity_covariance:
            scales = nn.Dense(
                self.num_dimensions * self.num_components, kernel_init=init
            )(inputs)
            scales = scales.reshape(shape)
        else:
            scales = jnp.ones_like(locs)

        components = tfp.distributions.MultivariateNormalDiag(
            loc=locs, scale_diag=scales
        )
        mixture = tfp.distributions.Categorical(logits=logits)
        return tfp.distributions.MixtureSameFamily(
            mixture_distribution=mixture, components_distribution=components
        )


class DynamicsNetwork(nn.Module):
    hidden_layer_sizes: Tuple[int, ...]
    output_size: int
    omit_input_dynamics_dim: int = 2
    identity_covariance: bool = True
    initializer: Optional[initializers.Initializer] = None

    @nn.compact
    def __call__(
        self, obs: StateDescriptor, skill: Skill, target: StateDescriptor
    ) -> jnp.ndarray:
        if self.initializer is None:
            init = initializers.variance_scaling(1.0, "fan_in", "uniform")
        else:
            init = self.initializer

        distribution = GaussianMixture(
            self.output_size,
            num_components=4,
            reinterpreted_batch_ndims=None,
            identity_covariance=self.identity_covariance,
            initializer=init,
        )

        obs = obs[:, self.omit_input_dynamics_dim :]
        obs = jnp.concatenate((obs, skill), axis=1)

        x = MLP(
            layer_sizes=self.hidden_layer_sizes,
            kernel_init=init,
            activation=nn.relu,
            final_activation=nn.relu,
        )(obs)

        dist = distribution(x)
        return dist.log_prob(target)


class Actor(nn.Module):
    action_size: int
    hidden_layer_sizes: Tuple[int, ...]

    @nn.compact
    def __call__(self, obs: Observation) -> jnp.ndarray:
        init = initializers.variance_scaling(1.0, "fan_in", "uniform")

        return MLP(
            layer_sizes=self.hidden_layer_sizes + (2 * self.action_size,),
            kernel_init=init,
            activation=nn.relu,
        )(obs)


class Critic(nn.Module):
    hidden_layer_sizes: Tuple[int, ...]

    @nn.compact
    def __call__(self, obs: Observation, action: Action) -> jnp.ndarray:
        init = initializers.variance_scaling(1.0, "fan_in", "uniform")
        input_ = jnp.concatenate([obs, action], axis=-1)

        value_1 = MLP(
            layer_sizes=self.hidden_layer_sizes + (1,),
            kernel_init=init,
            activation=nn.relu,
        )(input_)

        value_2 = MLP(
            layer_sizes=self.hidden_layer_sizes + (1,),
            kernel_init=init,
            activation=nn.relu,
        )(input_)

        return jnp.concatenate([value_1, value_2], axis=-1)


def make_dads_networks(
    action_size: int,
    descriptor_size: int,
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256),
    policy_hidden_layer_size: Tuple[int, ...] = (256, 256),
    omit_input_dynamics_dim: int = 2,
    identity_covariance: bool = True,
    dynamics_initializer: Optional[initializers.Initializer] = None,
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    policy = Actor(action_size, policy_hidden_layer_size)
    critic = Critic(critic_hidden_layer_size)
    dynamics = DynamicsNetwork(
        critic_hidden_layer_size,
        descriptor_size,
        omit_input_dynamics_dim=omit_input_dynamics_dim,
        identity_covariance=identity_covariance,
        initializer=dynamics_initializer,
    )

    return policy, critic, dynamics
