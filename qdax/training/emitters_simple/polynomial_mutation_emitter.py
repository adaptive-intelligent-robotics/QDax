import functools

import jax
import jax.numpy as jnp


def polynomial_mutation_emitter(repertoire, key, population_size: int, eta_m: float):

    key_selection, key_variation = jax.random.split(key, 2)

    # SELECTION #
    idx_p1 = jax.random.randint(
        key_selection, shape=(population_size,), minval=0, maxval=repertoire.num_indivs
    )
    tot_indivs = repertoire.fitness.ravel().shape[0]
    indexes = jnp.argwhere(
        jnp.logical_not(jnp.isnan(repertoire.fitness)), size=tot_indivs
    )
    indexes = jnp.transpose(indexes, axes=(1, 0))
    indiv_indices = jnp.array(
        jnp.ravel_multi_index(indexes, repertoire.fitness.shape, mode="clip")
    ).astype(int)

    idx_p1 = indiv_indices.at[idx_p1].get()
    pparams = jax.tree_map(lambda x: x.at[idx_p1].get(), repertoire.archive)

    # MUTATION #
    # eta_m = 5.0
    num_vars = len(jax.tree_leaves(pparams))
    treedef = jax.tree_structure(pparams)

    all_keys_a = jax.random.split(key_variation, num_vars)

    r = jax.tree_multimap(
        lambda g, k: jax.random.uniform(k, shape=g.shape, dtype=g.dtype),
        pparams,
        jax.tree_unflatten(treedef, all_keys_a),
    )

    delta = jax.tree_map(
        lambda g: jnp.where(
            g < 0.5,
            x=jnp.power(2.0 * g, 1.0 / (eta_m + 1.0)) - 1.0,
            y=1 - jnp.power(2.0 * (1.0 - g), 1.0 / (eta_m + 1.0)),
        ),
        r,
    )

    new_params = jax.tree_multimap(lambda p, d: p + d, pparams, delta)

    return new_params
