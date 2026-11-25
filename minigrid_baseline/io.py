import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .config import ROOT


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def write_csv(path, rows, append=False):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not append or not path.exists() or path.stat().st_size == 0
    with path.open("a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if header:
            writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def policy_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@contextmanager
def isolated_rng():
    py, np_state, th = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(th)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def metadata():
    # Public, software-only provenance. No host, user, scheduler or hardware inventory.
    names = ("torch", "stable-baselines3", "minigrid", "gymnasium", "numpy", "PyYAML")
    return dict(python=platform.python_version(), packages={n: importlib.metadata.version(n) for n in names})
