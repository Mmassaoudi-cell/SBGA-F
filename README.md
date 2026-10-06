# SBGA-F: Sparse Bidirectional Graph Attention with Focal Loss

Source code for **SBGA-F**, a graph-based intrusion detection model for
multiclass IoT/IIoT attack traffic, combining:

- a temporal encoder (LSTM + additive attention pooling) over per-node
  packet/flow windows,
- **learned, sparse, top-k bidirectional graph attention** (separate
  in-neighbor and out-neighbor attention heads) over a probabilistic
  device-communication graph,
- **topology-agnostic structural/positional node features** (in-degree,
  out-degree, PageRank, clustering coefficient) instead of node-identity
  embeddings, so the model can generalize zero-shot to device graphs of a
  different size than the one it was trained on,
- a class-balanced **multiclass focal loss**, for robustness to the heavy
  class imbalance typical of real IoT attack datasets.

An extension, **SBGA-F-Stack**, combines SBGA-F's learned representation
with a gradient-boosted-tree base learner through a logistic-regression
meta-learner, closing the same-distribution accuracy gap against boosting
baselines while retaining SBGA-F's zero-shot cross-topology transfer.

This repository contains the model and experiment source code for
independent reproduction. It intentionally does **not** include the
dataset, generated result tables, figures, or the manuscript — only the
code needed to run the full pipeline end-to-end on your own copy of the
data.

## Environment

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended; CPU-only execution works but
deep-learning/boosting training will be considerably slower.

## Data

The pipeline expects the public **Edge-IIoTset** dataset
(Ferrag et al., 2022), specifically `DNN-EdgeIIoT-dataset.csv`, at the path
configured at the top of `scripts/01_prepare_data.py` (edit the `SRC`
constant to point at your local copy). The dataset is not redistributed in
this repository.

## Repository structure

```
requirements.txt        Python environment
FINAL_MODEL_CONFIG.yaml Frozen SBGA-F architecture, hyperparameters, and
                         training/evaluation protocol
scripts/
  01_prepare_data.py        Cleaning, leakage-safe temporal split, graph
                             construction, packet/flow/fusion feature tables
  02_reproduce_source_method.py   Reproduction of the source-paper baseline
                                    (BiRAGNN) and its benchmark suite
  03_make_reproduction_report.py  Reproduction-report generation
  04_candidate_screening.py       Validation-only screening of candidate
                                    hybrid architectures (incl. SBGA-F)
  05_classical_benchmarks.py      Classical/boosting benchmark suite
  06_deep_benchmarks_multiclass.py  Multiclass retraining of deep/graph
                                      benchmark architectures
  07_tune_finalists.py            Hyperparameter tuning (Optuna) of the
                                    finalist candidates
  08_final_test_evaluation.py     Final test-set evaluation: SBGA-F vs. all
                                    benchmarks, multi-seed
  09_ablation.py                  Ablation study
  10_robustness_generalization.py Robustness / cross-topology transfer
  10b_robustness_baseline_comparison.py  Robustness baseline comparison
  11_statistics_wtl.py            Statistical testing (Wilcoxon + Holm),
                                    win/tie/loss
  12_generate_figures.py          Figure generation from result CSVs
  13_stacked_model.py             SBGA-F-Stack (stacking ensemble)
  14_transfer_ablation.py         Transfer-specific ablation
  15_stack_transfer_check.py      SBGA-F-Stack cross-topology transfer check
  16_stack_robustness_check.py    SBGA-F-Stack robustness check
  candidate_models.py    Candidate hybrid architectures, incl. SBGA-F
                          (`CandidateB_SBGAF`)
  models.py               Benchmark model architectures
  ft_transformer.py       FT-Transformer benchmark architecture
  common.py               Shared data-loading, windowing, and graph utilities
```

## Reproduction order

```bash
python scripts/01_prepare_data.py
python scripts/02_reproduce_source_method.py
python scripts/03_make_reproduction_report.py
python scripts/04_candidate_screening.py
python scripts/05_classical_benchmarks.py
python scripts/06_deep_benchmarks_multiclass.py
python scripts/07_tune_finalists.py
python scripts/08_final_test_evaluation.py
python scripts/09_ablation.py
python scripts/10_robustness_generalization.py
python scripts/10b_robustness_baseline_comparison.py
python scripts/11_statistics_wtl.py
python scripts/12_generate_figures.py
python scripts/13_stacked_model.py
python scripts/14_transfer_ablation.py
python scripts/15_stack_transfer_check.py
python scripts/16_stack_robustness_check.py
```

All random seeds are recorded in each script and in `FINAL_MODEL_CONFIG.yaml`.

## License

MIT (see `LICENSE`).
