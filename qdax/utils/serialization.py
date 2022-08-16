import pickle
from pathlib import Path
from typing import Any, Callable, Union

import jax
import jax.numpy as jnp

from qdax.types import Genotype

SUFFIX_PICKLE = ".pickle"


def pickle_save(data: Any, path: Union[str, Path], overwrite: bool = False) -> None:
    path = Path(path)
    if path.suffix != SUFFIX_PICKLE:
        path = path.with_suffix(SUFFIX_PICKLE)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f"File {path} already exists.")
    with open(path, "wb") as file:
        pickle.dump(data, file)


def pickle_load(path: Union[str, Path]) -> Any:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    if path.suffix != SUFFIX_PICKLE:
        raise ValueError(f"Not a {SUFFIX_PICKLE} file: {path}")
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


def get_default_genotype_reconstruction_fn(
    batch_genotypes: Genotype,
) -> Callable[[jnp.ndarray], Genotype]:
    one_genotype = jax.tree_map(lambda x: x[0], batch_genotypes)
    shapes_tree = jax.tree_map(lambda x: x.shape, one_genotype)
    list_lengths, tree_structure = jax.tree_flatten(
        jax.tree_map(lambda x: jnp.prod(x), shapes_tree)
    )
    list_indexes = []
    current_index = 0
    for length in list_lengths:
        current_index += length
        list_indexes.append(current_index)
    list_indexes.pop(-1)

    def _genotype_reconstruction_fn(_genotype_array: jnp.ndarray) -> Genotype:
        list_sub_genotypes = jnp.split(_genotype_array, list_indexes)
        tree_genotypes_flat = jax.tree_unflatten(tree_structure, list_sub_genotypes)
        tree_genotypes = jax.tree_map(
            lambda _arr, _shape: jnp.reshape(_arr, _shape),
            tree_genotypes_flat,
            shapes_tree,
        )
        return tree_genotypes

    return _genotype_reconstruction_fn
