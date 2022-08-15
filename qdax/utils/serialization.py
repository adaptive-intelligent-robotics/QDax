import pickle
from pathlib import Path
from typing import Any, Union

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
