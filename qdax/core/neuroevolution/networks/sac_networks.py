from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from qdax.types import Action, Observation


def make_sac_networks(
    action_size: int,
    hidden_layer_sizes: Tuple[int, ...] = (256, 256),
) -> Tuple[hk.Transformed, hk.Transformed]:
    """Creates networks used in SAC.

    Args:
        action_size: the size of the environment's action space
        hidden_layer_sizes: the number of neurons for hidden layers.
            Defaults to (256, 256).

    Returns:
        the policy network
        the critic network
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

    policy = hk.without_apply_rng(hk.transform(_actor_fn))
    critic = hk.without_apply_rng(hk.transform(_critic_fn))

    return policy, critic
