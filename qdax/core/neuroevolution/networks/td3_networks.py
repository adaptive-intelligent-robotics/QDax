""" Implements a function to create neural networks for the TD3 algorithm."""

from typing import Tuple

from jax import numpy as jnp

from qdax.core.neuroevolution.networks.networks import MLP, QModule


def make_td3_networks(
    action_size: int,
    critic_hidden_layer_sizes: Tuple[int, ...],
    policy_hidden_layer_sizes: Tuple[int, ...],
) -> Tuple[MLP, QModule]:
    """Creates networks used by the TD3 agent.

    Args:
        action_size: Size the action array used to interact with the environment.
        critic_hidden_layer_sizes: Number of layers and units per layer used in the
            neural network defining the critic.
        policy_hidden_layer_sizes: Number of layers and units per layer used in the
            neural network defining the policy.

    Returns:
        The neural network defining the policy and the module defininf the critic.
        This module contains two neural networks.
    """

    # Instantiate policy and critics networks
    policy_layer_sizes = policy_hidden_layer_sizes + (action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        final_activation=jnp.tanh,
    )
    q_network = QModule(n_critics=2, hidden_layer_sizes=critic_hidden_layer_sizes)

    return (policy_network, q_network)
