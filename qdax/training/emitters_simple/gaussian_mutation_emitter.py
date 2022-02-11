import functools

import jax
import jax.numpy as jnp


def gaussian_mutation_emitter(repertoire,
                              key, 
                              population_size: int,
                              mutation_power: float):

    key_selection, key_variation = jax.random.split(key, 2)

    # SELECTION #
    idx_p1 = jax.random.randint(key_selection, shape=(population_size,), minval=0, maxval=repertoire.num_indivs)
    tot_indivs = repertoire.fitness.ravel().shape[0]
    indexes = jnp.argwhere(jnp.logical_not(jnp.isnan(repertoire.fitness)), size = tot_indivs)
    indexes = jnp.transpose(indexes, axes=(1, 0))
    indiv_indices = jnp.array(jnp.ravel_multi_index(indexes, repertoire.fitness.shape, mode='clip')).astype(int)

    idx_p1 = indiv_indices.at[idx_p1].get()
    pparams = jax.tree_map(lambda x: x.at[idx_p1].get(),repertoire.archive)

    # VARIATION - MUTATION #
    num_vars = len(jax.tree_leaves(pparams))
    treedef = jax.tree_structure(pparams)
    all_keys = jax.random.split(key_variation, num=num_vars)

    # Gaussian noise
    noise = jax.tree_multimap(
        lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype), pparams,
        jax.tree_unflatten(treedef, all_keys))

    # antithetic - could add noise in positive and negative direction
    params_with_noise = jax.tree_multimap(lambda g, n: g + n * mutation_power, pparams, noise)
    anit_params_with_noise = jax.tree_multimap(lambda g, n: g - n * mutation_power, pparams, noise)

    # standard GA just returns params_with_noise so we use that
    #return params_with_noise, anit_params_with_noise, noise
    return params_with_noise




    