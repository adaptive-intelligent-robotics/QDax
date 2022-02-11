import functools

import jax
import jax.numpy as jnp


def _apply_variation_iso_dd(iso_sigma, line_sigma, pparams_p1, pparams_p2, key):
  num_vars = len(jax.tree_leaves(pparams_p1))

  treedef = jax.tree_structure(pparams_p1)

  key_a, key_b = jax.random.split(key, 2)

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


def get_apply_variation_iso_dd_fn(iso_sigma, line_sigma):
  return functools.partial(
    _apply_variation_iso_dd,
    iso_sigma,
    line_sigma,
  )


def _get_indexes(population_size, number_batches_to_select, repertoire, key):
  keys_selection_list = jax.random.split(key, number_batches_to_select)

  selected_indexes_list = [
    jax.random.randint(key_selection, shape=(population_size,), minval=0, maxval=repertoire.num_indivs)
    for key_selection in keys_selection_list
  ]

  tot_indivs = repertoire.fitness.ravel().shape[0]

  indexes = jnp.argwhere(jnp.logical_not(jnp.isnan(repertoire.fitness)), size=tot_indivs)

  indexes = jnp.transpose(indexes, axes=(1, 0))

  indiv_indices = jnp.array(jnp.ravel_multi_index(indexes, repertoire.fitness.shape, mode='clip')).astype(int)

  return [
    indiv_indices.at[idx].get()
    for idx in selected_indexes_list
  ]


def _select_from_repertoire(get_indexes_fn, repertoire, key):
  idx_list = get_indexes_fn(repertoire, key)

  return [
    jax.tree_map(lambda x: x.at[idx_p].get(), repertoire.archive)
    for idx_p in idx_list
  ]


def get_select_from_repertoire_fn(population_size, number_batches_to_select):
  get_indexes_fn = functools.partial(_get_indexes,
                                     population_size,
                                     number_batches_to_select)
  select_from_repertoire_fn = functools.partial(_select_from_repertoire, get_indexes_fn)
  return jax.jit(select_from_repertoire_fn)


def _emit_individuals(selector_fn, variation_fn, repertoire, key):
  key_selection, key_variation = jax.random.split(key, 2)
  selected_params_list = selector_fn(repertoire, key_selection)
  return variation_fn(*selected_params_list, key_variation)


def get_emitter_fn(selector_fn, variation_fn):
  return functools.partial(
    _emit_individuals,
    selector_fn,
    variation_fn,
  )


def get_emitter_iso_line_dd(population_size: int, iso_sigma: float, line_sigma: float):
  return get_emitter_fn(selector_fn=get_select_from_repertoire_fn(population_size=population_size,
                                                                  number_batches_to_select=2),
                        variation_fn=get_apply_variation_iso_dd_fn(iso_sigma=iso_sigma,
                                                                   line_sigma=line_sigma)
                        )
