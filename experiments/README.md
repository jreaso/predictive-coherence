# Experiments

The code to reproduce the experiments in *Measuring Coherence in Predictive Models* (NeurIPS 2026). `pc_experiments` holds the machinery, and each experiment's design lives in its scripts.


## Setup

With [uv](https://docs.astral.sh/uv/) and Python 3.12 or later, from the repository root:

```bash
git clone https://github.com/jreaso/predictive-coherence && cd predictive-coherence
uv sync  # the library, the experiments and the dev tools
```

Every command below is run from the repository root.

**Data.** The tabular experiment downloads its eight datasets from OpenML through scikit-learn on first use, and caches them in `experiments/data/openml/`. The OpenML ids are in `experiments/src/pc_experiments/tabular/datasets.py` and `experiments/tables/tabular_datasets.csv`. Every other experiment generates synthetic data.

**TabPFN weights.** The tabular experiment uses TabPFN v2.5, whose weights are gated. On first use the `tabpfn` package asks, in the browser, for a one-time acceptance of their licence, then downloads weights to cache. To use an existing copy, point `TABPFN_MODEL_CACHE_DIR` at the directory with the weights.


## Reproducing experiments

Each experiment has compute scripts, which write results to `experiments/outputs/<experiment>/`, and a `report.py`, which reads them and creates the plots and tables (writing to `experiments/figures/` and `experiments/tables/`). Optionally add the `--png` flag to the report script to add png plot outputs. The compute scripts keep finished work unless given `--overwrite`, except `blr_acquisition/run.py` and `tabular/run_checks.py`, which always replace theirs. Results files store ‖Δ‖² under the key `c`, an earlier name for it.

Run each experiment's commands in the order given.

**Figure 1 (rank inversion).**

```bash
# Figure 1 (no compute script)
uv run experiments/scripts/rank_inversion/report.py
```

**Experiment 1 (sparse GPs): Figure 2 and Tables 1–2.**

```bash
# Train the incTNP
uv run experiments/scripts/synthetic_gp/train_inctnp.py
# Run the sparse GPs and the incTNP
uv run experiments/scripts/synthetic_gp/run.py
# Figure 2 and Tables 1-2
uv run experiments/scripts/synthetic_gp/report.py
```

**Experiment 2 (real tabular datasets): Tables 3–7.**

```bash
# Tune each model for prediction
uv run experiments/scripts/tabular/tune.py
# Run every model except TabPFN
uv run experiments/scripts/tabular/run.py
# Run TabPFN, which has its own script because it is slow
uv run experiments/scripts/tabular/run_tabpfn.py
# Run the three checks
uv run experiments/scripts/tabular/run_checks.py
# Tables 3-7
uv run experiments/scripts/tabular/report.py
```

**Experiment 3 (acquisition within mean-field BLR): Figure 3 and Tables 8–9.**

```bash
# Run the baseline and the four robustness settings
uv run experiments/scripts/blr_acquisition/run.py
# Figure 3 and Tables 8-9
uv run experiments/scripts/blr_acquisition/report.py
```

**Experiment 4 (acquisition on a neural testbed): Figure 4 and Table 10.**

```bash
# Tune the model families and train the incTNP
uv run experiments/scripts/neural_testbed/tune.py
uv run experiments/scripts/neural_testbed/train_inctnp.py
# Run the testbed
uv run experiments/scripts/neural_testbed/run.py
# Figure 4 and Table 10
uv run experiments/scripts/neural_testbed/report.py
```


## Status of the results

The paper's results were computed during development, at earlier versions of this code, and have not been rerun at this version. The code that computes them has not changed since in anything that affects them, and the figures in `experiments/figures/` and tables in `experiments/tables/` were regenerated from those results by the commands above. Rerunning should reproduce them, but has not been explicitly tested.


## Credits

`experiments/src/pc_experiments/inctnp/` adapts the network from the authors' implementation of
[Incremental Transformer Neural Processes](https://arxiv.org/abs/2602.18955)
([code](https://github.com/philipmortimer/incTNP-code)). Each file names what it took, and the
licence is in `experiments/src/pc_experiments/inctnp/LICENSE`.

Parts of this code were written with the help of AI coding assistants and reviewed by the
authors.
