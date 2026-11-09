"""TabPFN, with and without its shipped preprocessing. It takes in data by adding it to the
context, and draws a block one input at a time, each added to the context before the next."""

from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from functools import cache
from typing import Any, Self

import numpy as np
import torch
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.settings import settings
from torch import Tensor
from torch.distributions import Categorical, Distribution

from pc_experiments.tabular.predictives import Bars
from predictive_coherence import PredictiveModel

# Switched off for the network alone: data-dependent transforms of inputs and targets, a hash
# feature, and reordering of features and classes.
NETWORK_ONLY = {
    "PREPROCESS_TRANSFORMS": [{"name": "none", "categorical_name": "numeric"}],
    "REGRESSION_Y_PREPROCESS_TRANSFORMS": (None,),
    "POLYNOMIAL_FEATURES": "no",
    "FINGERPRINT_FEATURE": False,
    "FEATURE_SHIFT_METHOD": None,
    "CLASS_SHIFT_METHOD": None,
}
# The release whose weights both arms use.
VERSION = ModelVersion.V2_5
ROUNDS_PER_CALL = 32
# Below this many context rows the package does not infer which inputs are categorical.
MIN_CONTEXT = 100


@cache
def _network(classify: bool, preprocessing: bool) -> Any:
    """One forward pass at temperature 1, so that the two arms differ in preprocessing alone."""
    # Contexts reach 1250 rows, past the 1000 the package otherwise refuses on a CPU.
    settings.tabpfn.allow_cpu_large_dataset = True
    estimator = TabPFNClassifier if classify else TabPFNRegressor
    return estimator.create_default_for_version(
        VERSION,
        n_estimators=1,
        softmax_temperature=1.0,
        inference_config=None if preprocessing else NETWORK_ONLY,
        device="cuda" if torch.cuda.is_available() else "cpu",  # slower on MPS than on the CPU
        random_state=0,
    )


@contextmanager
def _float32() -> Generator[None]:
    """The package builds tensors in torch's default dtype and breaks if that is float64."""
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(dtype)


class TabPFN(PredictiveModel):
    """TabPFN given a context `(n, d)`, `(n,)`, with or without its shipped preprocessing."""

    def __init__(self, x: Tensor, y: Tensor, num_classes: int, preprocessing: bool) -> None:
        self.x, self.y = x, y
        self.num_classes, self.preprocessing = num_classes, preprocessing

    def predict(self, x: Tensor) -> Distribution:
        return self._predict([self.x.numpy()], [self.y.numpy()], [x.numpy()])[0]

    def sample(self, x: Tensor, num_samples: int) -> Tensor:
        return self.predict(x).sample((num_samples,))

    def joint_sample(self, x: Tensor) -> Tensor:
        """Outputs at `(n, d)` drawn one input at a time, in the order given, not a random one."""
        model, drawn = self, []
        for row in x.unsqueeze(1):
            drawn.append(model.sample(row, 1)[0])
            model = model.update(row, drawn[-1])
        return torch.cat(drawn)

    def update(self, x: Tensor, y: Tensor) -> Self:
        return type(self)(
            torch.cat([self.x, x]), torch.cat([self.y, y]), self.num_classes, self.preprocessing
        )

    def updated(self, blocks: Tensor, x: Tensor) -> Iterator[Distribution]:
        """`update(block, joint_sample(block)).predict(x)` for each block of `(R, k, d)`, in
        distribution, with the rounds batched."""
        for chunk in (rounds.numpy() for rounds in blocks.split(ROUNDS_PER_CALL)):
            contexts = [self.x.numpy()] * len(chunk)
            outputs = [self.y.numpy()] * len(chunk)
            for step in range(chunk.shape[1]):
                inputs = list(chunk[:, step : step + 1])
                drawn = [p.sample().numpy() for p in self._predict(contexts, outputs, inputs)]
                contexts = [np.concatenate(pair) for pair in zip(contexts, inputs, strict=True)]
                outputs = [np.concatenate(pair) for pair in zip(outputs, drawn, strict=True)]
            yield from self._predict(contexts, outputs, [x.numpy()] * len(chunk))

    def _predict(
        self,
        contexts: Sequence[np.ndarray],
        outputs: Sequence[np.ndarray],
        inputs: Sequence[np.ndarray],
    ) -> list[Distribution]:
        with _float32():
            network = _network(bool(self.num_classes), self.preprocessing)
            if not self.num_classes:
                full = network.predict_batched(contexts, outputs, inputs, output_type="full")
                return [Bars(out["logits"].cpu(), out["criterion"].borders.cpu()) for out in full]
            seen_probs = network.predict_proba_batched(contexts, outputs, inputs)
        # A class absent from the context has no column of its own, and probability zero.
        predictives: list[Distribution] = []
        for labels, seen in zip(outputs, seen_probs, strict=True):
            probs = torch.zeros(len(seen), self.num_classes)
            probs[:, torch.tensor(np.unique(labels))] = torch.tensor(seen).to(probs)
            predictives.append(Categorical(probs=probs))
        return predictives
