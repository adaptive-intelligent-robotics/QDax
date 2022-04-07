from typing import Any, Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class QModule(nn.Module):
    """Q Module."""

    hidden_layer_sizes: Tuple[int, ...]
    n_critics: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
        hidden = jnp.concatenate([obs, actions], axis=-1)
        res = []
        for _ in range(self.n_critics):
            q = MLP(
                layer_sizes=self.hidden_layer_sizes + (1,),
                activation=nn.relu,
                kernel_init=jax.nn.initializers.lecun_uniform(),
            )(hidden)
            res.append(q)
        return jnp.concatenate(res, axis=-1)


class MLP(nn.Module):
    """MLP module."""

    layer_sizes: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    final_activation: Callable[[jnp.ndarray], jnp.ndarray] = None
    bias: bool = True
    kernel_init_final: Optional[Callable[..., Any]] = None

    @nn.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):

            if i != len(self.layer_sizes) - 1:
                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
                )(hidden)
                hidden = self.activation(hidden)

            else:
                if self.kernel_init_final is not None:
                    kernel_init = self.kernel_init_final
                else:
                    kernel_init = self.kernel_init

                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=kernel_init,
                    use_bias=self.bias,
                )(hidden)

                if self.final_activation is not None:
                    hidden = self.final_activation(hidden)

        return hidden
