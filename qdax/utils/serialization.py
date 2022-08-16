import pickle
from pathlib import Path
from typing import Any, Callable, Union

import jax
import jax.numpy as jnp

from qdax.types import Genotype

SUFFIX = ".pickle"


def pickle_save(data: Any, path: Union[str, Path], overwrite: bool = False) -> None:
    path = Path(path)
    if path.suffix != SUFFIX:
        path = path.with_suffix(SUFFIX)
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
    if path.suffix != SUFFIX:
        raise ValueError(f"Not a {SUFFIX} file: {path}")
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


def get_default_genotype_reconstruction_fn(
    batch_genotypes: Genotype,
) -> Callable[[jnp.ndarray], Genotype]:
    one_genotype = jax.tree_map(lambda x: x[0], batch_genotypes)
    shapes = jax.tree_map(lambda x: x.shape, one_genotype)
    list_lengths, tree_structure = jax.tree_flatten(
        jax.tree_map(lambda x: jnp.prod(x), shapes)
    )
    list_indexes = []
    current_index = 0
    for length in list_lengths:
        current_index += length
        list_indexes.append(current_index)
    list_indexes.pop(-1)

    def _genotype_reconstruction_fn(_genotype_array: jnp.ndarray) -> Genotype:

        list_sub_genotypes = jnp.split(_genotype_array, list_indexes)
        tree_genotypes = jax.tree_unflatten(tree_structure, list_sub_genotypes)
        return tree_genotypes

    return _genotype_reconstruction_fn
