"""Exclusive outputs and temporary staging for completed result publication."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np


def require_new_directory(path: str | Path) -> Path:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists; choose a new directory: {path}")
    return path


@contextmanager
def new_directory(path: str | Path):
    """Compute inside an owned sibling; a failed run leaves no result directory."""
    destination = require_new_directory(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.pending-", dir=destination.parent))
    try:
        yield staging
        require_new_directory(destination)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def require_new_prediction(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("prediction output must end in .npz")
    for target in (path, path.with_suffix(".json")):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"output already exists; choose new prediction paths: {target}")
    return path


def write_predictions(path: str | Path, arrays: dict, report: dict) -> None:
    """Create both files exclusively; clean up only files this call created."""
    destination = require_new_prediction(path)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with destination.open("xb") as handle:
            created.append(destination)
            np.savez_compressed(handle, **arrays)
        sidecar = destination.with_suffix(".json")
        with sidecar.open("x", encoding="utf-8") as handle:
            created.append(sidecar)
            handle.write(text)
    except BaseException:
        for target in reversed(created):
            target.unlink(missing_ok=True)
        raise
