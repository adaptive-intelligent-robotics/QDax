import pickle
import warnings
from pathlib import Path
from typing import Any, Callable, Union

import jax
import jax.flatten_util
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
    warnings.warn(
        "A pickle file is being loaded. "
        "Always ensure you trust the source of the file."
    )
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
    _, genotype_reconstruction_fn = jax.flatten_util.ravel_pytree(one_genotype)
    genotype_reconstruction_fn: Callable[[jnp.ndarray], Genotype]
    return genotype_reconstruction_fn
