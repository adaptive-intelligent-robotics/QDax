import jax
import jax.numpy as jnp


def rastrigin_scoring(x: jnp.ndarray):
    return -(10 * x.shape[-1] + jnp.sum((x+minval*0.4)**2 - 10 * jnp.cos(2 * jnp.pi * (x+minval*0.4))))

def clip(x: jnp.ndarray):
    return x*(x<=maxval)*(x>=+minval) + maxval/x*((x>maxval)+(x<+minval))

def _rastrigin_descriptor_1(x: jnp.ndarray):
    return jnp.mean(clip(x[:x.shape[0]//2]))

def _rastrigin_descriptor_2(x: jnp.ndarray):
    return jnp.mean(clip(x[x.shape[0]//2:]))

def rastrigin_descriptors(x: jnp.ndarray):
    return jnp.array([_rastrigin_descriptor_1(x), _rastrigin_descriptor_2(x)])