import functools

import jax
import jax.numpy as jnp

# could even just pass in the training state entirely instead of just the repertoire

def iso_dd_emitter(repertoire,
                   key, 
                   population_size: int,
                   iso_sigma: float = 0.01, 
                   line_sigma: float = 0.1):

    key_selection, key_variation = jax.random.split(key, 2)

    # SELECTION #
    key_select_p1, key_select_p2 = jax.random.split(key_selection, 2)
    idx_p1 = jax.random.randint(key_select_p1, shape=(population_size,), minval=0, maxval=repertoire.num_indivs)
    idx_p2 = jax.random.randint(key_select_p2, shape=(population_size,), minval=0, maxval=repertoire.num_indivs)
    tot_indivs = repertoire.fitness.ravel().shape[0]
    indexes = jnp.argwhere(jnp.logical_not(jnp.isnan(repertoire.fitness)), size = tot_indivs)
    indexes = jnp.transpose(indexes, axes=(1, 0))
    indiv_indices = jnp.array(jnp.ravel_multi_index(indexes, repertoire.fitness.shape, mode='clip')).astype(int)

    idx_p1 = indiv_indices.at[idx_p1].get()
    idx_p2 = indiv_indices.at[idx_p2].get()
    pparams_p1 = jax.tree_map(lambda x: x.at[idx_p1].get(),repertoire.archive)
    pparams_p2 = jax.tree_map(lambda x: x.at[idx_p2].get(),repertoire.archive)

    # VARIATION #
    num_vars = len(jax.tree_leaves(pparams_p1))
    treedef = jax.tree_structure(pparams_p1)
    key_a, key_b = jax.random.split(key_variation, 2)
    all_keys_a = jax.random.split(key_a, num_vars)
    all_keys_b = jax.random.split(key_b, num_vars)

    noise_a = jax.tree_multimap(
        lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype), pparams_p1,
        jax.tree_unflatten(treedef, all_keys_a))
    noise_b = jax.tree_multimap(
        lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype), pparams_p2,
        jax.tree_unflatten(treedef, all_keys_b))

    new_params = jax.tree_multimap(lambda x, y, a, b: x + a * iso_sigma + b * line_sigma * (x - y),
                                    pparams_p1, pparams_p2, noise_a, noise_b)

    return new_params



def create_iso_dd_fn(population_size: int,
                     iso_sigma: float = 0.01, 
                     line_sigma: float = 0.1):

    def iso_dd_fn(repertoire, key):
        key_selection, key_variation = jax.random.split(key, 2)
        # SELECTION #
        key_select_p1, key_select_p2 = jax.random.split(key_selection, 2)
        idx_p1 = jax.random.randint(key_select_p1, shape=(population_size,), minval=0, maxval=repertoire.num_indivs)
        idx_p2 = jax.random.randint(key_select_p2, shape=(population_size,), minval=0, maxval=repertoire.num_indivs)
        tot_indivs = repertoire.fitness.ravel().shape[0]
        indexes = jnp.argwhere(jnp.logical_not(jnp.isnan(repertoire.fitness)), size = tot_indivs)
        indexes = jnp.transpose(indexes, axes=(1, 0))
        indiv_indices = jnp.array(jnp.ravel_multi_index(indexes, repertoire.fitness.shape, mode='clip')).astype(int)

        idx_p1 = indiv_indices.at[idx_p1].get()
        idx_p2 = indiv_indices.at[idx_p2].get()
        pparams_p1 = jax.tree_map(lambda x: x.at[idx_p1].get(),repertoire.archive)
        pparams_p2 = jax.tree_map(lambda x: x.at[idx_p2].get(),repertoire.archive)

        # VARIATION #
        num_vars = len(jax.tree_leaves(pparams_p1))
        treedef = jax.tree_structure(pparams_p1)
        key_a, key_b = jax.random.split(key_variation, 2)
        all_keys_a = jax.random.split(key_a, num_vars)
        all_keys_b = jax.random.split(key_b, num_vars)

        noise_a = jax.tree_multimap(
            lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype), pparams_p1,
            jax.tree_unflatten(treedef, all_keys_a))
        noise_b = jax.tree_multimap(
            lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype), pparams_p2,
            jax.tree_unflatten(treedef, all_keys_b))

        new_params = jax.tree_multimap(lambda x, y, a, b: x + a * iso_sigma + b * line_sigma * (x - y),
                                        pparams_p1, pparams_p2, noise_a, noise_b)
        
        return new_params

    return iso_dd_fn