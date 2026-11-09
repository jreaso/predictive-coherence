"""Tabular datasets from OpenML, and how one is split into a task."""

import zlib
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from torch import Tensor

from pc_experiments.outputs import data_dir

# OpenML data ids.
REGRESSION = {
    "concrete": 44959,
    "california": 44977,
    "yacht": 42370,
}
CLASSIFICATION = {
    "australian": 40981,
    "cleveland": 194,
    "german": 31,
    "vehicle": 54,
    "wdbc": 1510,
}


@dataclass(frozen=True)
class Dataset:
    """A dataset's complete rows, inputs `(N, d)` one-hot encoded and outputs `(N,)`, real or
    labels. `num_classes` is 0 for regression."""

    name: str
    x: Tensor
    y: Tensor
    num_classes: int


def encode(name: str, inputs: pd.DataFrame, target: pd.Series, classify: bool) -> Dataset:
    """Drop rows with missing values, one-hot encode categorical inputs, and code the labels."""
    complete = (inputs.notna().all(axis=1) & target.notna()).to_numpy()
    x = torch.tensor(pd.get_dummies(inputs).to_numpy(dtype="float64")[complete])
    if not classify:
        return Dataset(name, x, torch.tensor(target.to_numpy(dtype="float64")[complete]), 0)
    labels = target.to_numpy()[complete]
    # Numbers are coded in their order, anything else in the order of its text.
    classes, codes = np.unique(labels if labels.dtype.kind in "iuf" else labels.astype(str),
                               return_inverse=True)  # fmt: skip
    return Dataset(name, x, torch.tensor(codes), len(classes))


@cache
def load(name: str) -> Dataset:
    """Fetch a dataset, or read it from `experiments/data/openml/` once it has been fetched."""
    classify = name in CLASSIFICATION
    data_id = CLASSIFICATION[name] if classify else REGRESSION[name]
    inputs, target = fetch_openml(
        data_id=data_id, as_frame=True, return_X_y=True, data_home=str(data_dir() / "openml")
    )
    return encode(name, inputs, target, classify)


@dataclass(frozen=True)
class Sizes:
    """Shares of a dataset's rows, each capped, for the training and test sets and the pool of
    the rest; a block is a share of the training set."""

    train_share: float
    train_cap: int
    test_share: float
    test_cap: int
    pool_cap: int
    block_share: float


@dataclass(frozen=True)
class Task:
    """One split of a dataset, standardised on its training set: training `(n, d)`, test
    `(M, d)` and pool `(P, d)`."""

    dataset: str
    seed: int
    x: Tensor
    y: Tensor
    x_test: Tensor
    y_test: Tensor
    x_pool: Tensor
    block_size: int
    num_classes: int

    def blocks(self, num_rounds: int) -> Tensor:
        """Each round's block from the pool, without replacement, `(R, block, d)`: the same for
        every arm."""
        generator = torch.Generator().manual_seed(checksum(f"{self.dataset}:{self.seed}:blocks"))
        rows = [
            torch.randperm(len(self.x_pool), generator=generator)[: self.block_size]
            for _ in range(num_rounds)
        ]
        return self.x_pool[torch.stack(rows)]


def split_rows(dataset: Dataset, sizes: Sizes, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    """The rows of split `seed`: training, test and pool, disjoint and uniform at random."""
    num_rows = len(dataset.x)
    num_train = min(int(sizes.train_share * num_rows), sizes.train_cap)
    num_test = min(int(sizes.test_share * num_rows), sizes.test_cap)
    num_pool = min(num_rows - num_train - num_test, sizes.pool_cap)
    generator = torch.Generator().manual_seed(checksum(f"{dataset.name}:{seed}"))
    order = torch.randperm(num_rows, generator=generator)
    train, test, pool = order[: num_train + num_test + num_pool].split(
        [num_train, num_test, num_pool]
    )
    return train, test, pool


def make_task(dataset: Dataset, sizes: Sizes, seed: int) -> Task:
    """Split `dataset` uniformly at random into a training set, test inputs and a pool."""
    train, test, pool = split_rows(dataset, sizes, seed)
    block_size = max(1, int(sizes.block_share * len(train)))
    if block_size > len(pool):
        raise ValueError(f"{dataset.name}: a block of {block_size} from a pool of {len(pool)}")

    x, y = dataset.x, dataset.y
    centre, scale = x[train].mean(0), x[train].std(0)
    # A column constant over the training set is centred but not scaled.
    x = (x - centre) / torch.where(scale > 0, scale, 1.0)
    if not dataset.num_classes:
        y = (y - y[train].mean()) / y[train].std()
    return Task(
        dataset.name, seed, x[train], y[train], x[test], y[test], x[pool], block_size,
        dataset.num_classes,
    )  # fmt: skip


def checksum(text: str) -> int:
    """A seed from a label. Not `hash`, which Python salts differently in every process."""
    return zlib.crc32(text.encode())
