from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp

from qdax.custom_types import Action, Observation


class MLP(nn.Module):
    features: Tuple[int, ...]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for feat in self.features[:-1]:
            x = nn.relu(
                nn.Dense(
                    feat,
                    kernel_init=nn.initializers.variance_scaling(
                        1.0, "fan_in", "uniform"
                    ),
                )(x)
            )
        return nn.Dense(
            self.features[-1],
            kernel_init=nn.initializers.variance_scaling(1.0, "fan_in", "uniform"),
        )(x)


class Actor(nn.Module):
    action_size: int
    hidden_layer_size: Tuple[int, ...]

    @nn.compact
    def __call__(self, obs: Observation) -> jnp.ndarray:
        return MLP(self.hidden_layer_size + (2 * self.action_size,))(obs)


class Critic(nn.Module):
    hidden_layer_size: Tuple[int, ...]

    @nn.compact
    def __call__(self, obs: Observation, action: Action) -> jnp.ndarray:
        input_ = jnp.concatenate([obs, action], axis=-1)
        value1 = MLP(self.hidden_layer_size + (1,))(input_)
        value2 = MLP(self.hidden_layer_size + (1,))(input_)
        return jnp.concatenate([value1, value2], axis=-1)


class Discriminator(nn.Module):
    num_skills: int
    hidden_layer_size: Tuple[int, ...]

    @nn.compact
    def __call__(self, obs: Observation) -> jnp.ndarray:
        return MLP(self.hidden_layer_size + (self.num_skills,))(obs)


def make_diayn_networks(
    action_size: int,
    num_skills: int,
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256),
    policy_hidden_layer_size: Tuple[int, ...] = (256, 256),
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Creates networks used in DIAYN.

    Args:
        action_size: the size of the environment's action space
        num_skills: the number of skills set
        critic_hidden_layer_size: the number of neurons for critic hidden layers.
        policy_hidden_layer_size: the number of neurons for policy hidden layers.

    Returns:
        the policy network
        the critic network
        the discriminator network
    """
    policy = Actor(action_size, policy_hidden_layer_size)
    critic = Critic(critic_hidden_layer_size)
    discriminator = Discriminator(num_skills, critic_hidden_layer_size)

    return policy, critic, discriminator
