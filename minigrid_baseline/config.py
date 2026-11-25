import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ENV = "MiniGrid-DoorKey-6x6-v0"
TEST_ENVS = (TRAIN_ENV, "MiniGrid-DoorKey-8x8-v0")


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path="configs/baseline.yaml"):
    with project_path(path).open() as f:
        return yaml.safe_load(f)


def load_episode_seeds(path="configs/eval_episode_seeds.json"):
    with project_path(path).open() as f:
        data = json.load(f)
    lists = [data[k] for k in ("validation_6x6", "test_6x6", "test_8x8")]
    for values in lists:
        if len(values) != 100 or len(set(values)) != 100:
            raise ValueError("Each evaluation split must contain exactly 100 unique seeds")
        if not all(type(v) is int and v >= 0 for v in values):
            raise ValueError("Episode seeds must be nonnegative integers")
    if any(set(lists[i]) & set(lists[j]) for i in range(3) for j in range(i)):
        raise ValueError("Validation/test episode seed lists must be disjoint")
    return data


def select_seeds(data, env_id, split, episodes=100):
    if env_id not in TEST_ENVS or split not in ("validation", "test"):
        raise ValueError("Unsupported environment or split")
    if split == "validation" and env_id != TRAIN_ENV:
        raise ValueError("Validation is defined only for 6x6")
    if not 1 <= episodes <= 100:
        raise ValueError("episodes must be between 1 and 100")
    size = "6x6" if env_id == TRAIN_ENV else "8x8"
    return data[f"{split}_{size}"][:episodes]

