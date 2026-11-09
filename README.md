# Measuring Coherence in Predictive Models

Jamie Reason, Vik Shirvaikar, Mark van der Wilk and Chris C. Holmes (NeurIPS 2026)

Sequential decision-making procedures such as active learning implicitly assume that a model’s current predictions agree with those it expects to make after updating on new data. We call this property coherence. Most models in use don’t update by exact conditioning, so they aren’t guaranteed to be coherent, and this can lead to worse decisions.

This repository provides the incoherence ratio, an MMD-based measure of how far any updating procedure departs from coherence, computed only from the procedure’s own predictions and updates, together with an anytime-valid test of exact coherence. In the paper, acquisition objectives that are equivalent for coherent models choose different data under every approximate model we study, and the incoherence ratio tracks how far this distorts them.

## Predictive coherence library

We provide `predictive_coherence`, a small library for measuring and testing the coherence of any predictive model.

```bash
pip install "predictive-coherence @ git+https://github.com/jreaso/predictive-coherence"
```

- `PredictiveModel` is the interface your model implements: `sample` from its predictive and `update` on new data. Optionally, `predict` can return a `torch.distributions` predictive, so embeddings are computed exactly rather than from samples (preferable when available).
- Kernels on outputs: `DiracKernel` for class labels and `RFFKernel` (random Fourier features of a Gaussian kernel) for real-valued outputs.
- `iter_rounds` simulates updates of a model on data drawn from its own predictive.
- `CoherenceEstimator` estimates ‖Δ‖², how far the expected future predictive is from the current one, and V, how much individual updates vary. It reports the incoherence ratio Ψ = ‖Δ‖² / V with a leave-one-round-out standard error.
- `SequentialTest` is an anytime-valid test of coherence. You can stop it whenever you like, and its error guarantee still holds.

### Usage

```python
from itertools import islice
from predictive_coherence import CoherenceEstimator, RFFKernel, SequentialTest, iter_rounds

# `model` implements PredictiveModel, updates draw outputs at x_design, Psi is measured at x_eval.
rounds = iter_rounds(model, RFFKernel(lengthscale=0.1), x_design, x_eval)
test, estimator = SequentialTest(), CoherenceEstimator()
for r in islice(rounds, 100):  # the test and the estimate share the same updates
    test.update(r.mu_p, r.mu_r)
    estimator.update(*r)
estimate = estimator.estimate()
print(estimate.psi, estimate.standard_error(), test.rejection_round(alpha=0.05))
```

The incoherence ratio Ψ is pooled over the evaluation inputs; `estimate.psi_per_point` localises it. For a model without a closed-form predictive, pass `num_samples` to `iter_rounds`.

## Experiments

The paper's experiments are in `experiments/`, and [`experiments/README.md`](experiments/README.md) says how to set them up and reproduce them.


## Citation

```bibtex
@inproceedings{reason2026measuring,
  title     = {Measuring Coherence in Predictive Models},
  author    = {Reason, Jamie and Shirvaikar, Vik and van der Wilk, Mark and Holmes, Chris C.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

For questions about the paper, contact Jamie Reason at [jamie.reason@reuben.ox.ac.uk](mailto:jamie.reason@reuben.ox.ac.uk).
