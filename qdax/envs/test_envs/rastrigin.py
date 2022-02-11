import jax
import jax.numpy as jnp

import math


@jax.vmap
def rastrigin(params):
    '''
    10d parameter
    2d BD
    '''
    x = params * 10 - 5 # scaling to [-5, 5]
    f = 10 * x.shape[0] + (x * x - 10 * jnp.cos(2 * math.pi * x)).sum()
    return -f, jnp.array([params[0], params[1]])


@jax.vmap
def sphere(params):
    '''
    10d parameter
    2d BD
    '''
    x = params * 10 - 5 # scaling to [-5, 5]
    f = (x * x).sum()
    return -f, jnp.array([params[0], params[1]])



def test_rastrigin():
    seed =10
    key = jax.random.PRNGKey(seed)

    param_size = 10
    batch_size = 200
    # generate a batch of params - params lie between 0 and 1 (10-D parameter)
    params_batch = jax.random.uniform(key, shape=(batch_size, param_size), minval=0, maxval=1)
    
    # evaluate params on rastrigin
    eval_scores, bd = rastrigin(params_batch)

    print(eval_scores.shape)
    print(bd.shape)

    print(eval_scores)
    print(bd)


# test_rastrigin()