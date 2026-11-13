# SBGA-F: Sparse Bidirectional Graph Attention with Focal Loss

Source code for the paper

> **Stacking Sparse Bidirectional Graph Attention with Gradient Boosted Trees
> for Multiclass Intrusion Detection in Internet of Things Networks**
> Mohamed Massaoudi and Maymouna Ez Eddin

**SBGA-F** is a multiclass intrusion detector for IoT/IIoT traffic that combines

- a temporal encoder (LSTM + attention pooling) over the ten most recent
  feature vectors of a device,
- learned, masked top-k attention over the prototypes of the strongest
  incoming and outgoing neighbors of the device in an event-transition graph,
- topology-derived node descriptors (in-degree, out-degree, PageRank,
  clustering) in place of node-identity embeddings, so that no parameter
  depends on the number of devices,
- a class-balanced multiclass focal loss.

**SBGA-F-Stack** fuses the class probabilities of SBGA-F and CatBoost through
a logistic-regression meta-learner fitted on validation data.

The repository contains the model and experiment code only. It does not
include datasets, result tables, figures, or the manuscript.

## Environment

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended. On some PyTorch/cuDNN builds the float32
cuDNN LSTM kernel is slow; setting the environment variable
`SBGAF_NATIVE_LSTM=1` selects PyTorch's native LSTM kernel (same model, same
precision) for the scripts that import `exp_common.py`.

## Data

Two public datasets are used. Neither is redistributed here.

| Dataset | File | Where the path is set |
|---|---|---|
| Edge-IIoTset (Ferrag et al., 2022) | `DNN-EdgeIIoT-dataset.csv` | `SRC` in `scripts/01_prepare_data.py` |
| TON_IoT network subset (Moustafa, 2021) | `train_test_network.csv` | `SRC` in `scripts/36_r2_ton.py` |

Edit the two `SRC` constants to point at your local copies.

## Repository structure

```
requirements.txt          Python environment
FINAL_MODEL_CONFIG.yaml   Frozen SBGA-F architecture, hyperparameters and protocol
config/tuned_baselines.json   Baseline hyperparameters selected with the same
                              tuning budget as SBGA-F (20 TPE trials)
scripts/
  Shared modules
    exp_common.py         Data loading, windows, graph construction, prototypes,
                          SBGA-F (SBGAFv2), training loops, stacking, perturbations
    common.py             Data-loading, windowing and graph utilities
    models.py             Benchmark architectures (FNN, RNN, CNN, Transformer,
                          GNN, TGNN, BiRAGNN)
    candidate_models.py   Focal loss, TCN encoder, candidate architectures
    ft_transformer.py     FT-Transformer benchmark

  Data preparation
    01_prepare_data.py    Cleaning, class-wise chronological split, sampling,
                          host assignment, flow aggregation, graphs (8/50/100 nodes)
    17_extract_aux_hosts.py   Host and timestamp columns aligned with the feature table

  Model development (screening and tuning)
    02 ... 16             Source-method reproduction, candidate screening,
                          benchmarks, Optuna tuning of SBGA-F, first evaluation

  Main experiments (ten seeds)
    18_rev_main.py        16-model benchmark, SBGA-F-Stack, perturbation suite
    19_rev_ablation.py    Component ablation (standalone)
    20_rev_graph_controls.py   Alternative graph estimators, interleaved stream
    21_rev_transfer.py, 21b_rev_transfer_tw.py   Parameter reuse across graph sizes
    23_rev_runtime.py     End-to-end runtime by stage
    22_rev_analysis.py, 25_rev_figures.py        Tables and figures

  Controls and replication (ten seeds)
    31_r2_ablation.py     Ablation at the standalone and stack levels
                          (attention directions, neighborhood size, descriptors,
                          aggregation, encoder, loss)
    32_r2_interleaved_causal.py   Class-interleaved stream; causal flow features
    33_r2_transfer.py     Reuse with limited target data, temporal-only and
                          node-indexed baselines, withheld devices
    34_r2_robust.py       Degraded/stale graph context, augmented tree baselines,
                          repeated perturbation draws, FGSM perturbation
    35_r2_tune.py         Equal tuning budget for the baselines; tuned evaluation
    36_r2_ton.py          TON_IoT preprocessing, benchmark, stacks, graph-size reuse
    37_r2_audit.py        Window label purity, duplicates, device-identity
                          classifier, stacking controls, per-class metrics,
                          bootstrap and McNemar tests
    38_r2_memory.py       Model size, GPU memory, CPU-only latency
    41_r2_ton_audit.py    Device-identity and per-class audit for TON_IoT
    39_r2_analysis.py, 40_r2_figures.py, 42_r2_tables.py   Statistics, figures, tables
```

## Reproduction order

Seeds: `42 123 2024 7 99 555 777 2025 31 88`.

```bash
# data
python scripts/01_prepare_data.py
python scripts/17_extract_aux_hosts.py

# main experiments (pass the seeds as arguments)
python scripts/18_rev_main.py 42 123 2024 7 99 555 777 2025 31 88
python scripts/19_rev_ablation.py 42 123 2024 7 99
python scripts/20_rev_graph_controls.py 42 123 2024 7 99
python scripts/21_rev_transfer.py 42 123 2024 7 99 555 777 2025 31 88
python scripts/21b_rev_transfer_tw.py 42 123 2024 7 99 555 777 2025 31 88
python scripts/23_rev_runtime.py
python scripts/22_rev_analysis.py
python scripts/25_rev_figures.py

# controls and replication (require the probabilities written by 18_rev_main.py)
python scripts/31_r2_ablation.py 42 123 2024 7 99 555 777 2025 31 88
python scripts/32_r2_interleaved_causal.py interleaved 42 123 2024 7 99 555 777 2025 31 88
python scripts/32_r2_interleaved_causal.py causal 42 123 2024 7 99 555 777 2025 31 88
python scripts/33_r2_transfer.py 42 123 2024 7 99 555 777 2025 31 88
python scripts/34_r2_robust.py 42 123 2024 7 99 555 777 2025 31 88
python scripts/35_r2_tune.py tune CatBoost LightGBM XGBoost RandomForest BiRAGNN RNN
python scripts/35_r2_tune.py eval 42 123 2024 7 99 555 777 2025 31 88
python scripts/36_r2_ton.py 42 123 2024 7 99 555 777 2025 31 88
python scripts/37_r2_audit.py
python scripts/41_r2_ton_audit.py
python scripts/38_r2_memory.py
python scripts/39_r2_analysis.py
python scripts/40_r2_figures.py
python scripts/42_r2_tables.py
```

Scripts `02`–`16` document the model-development stage (screening, tuning and
the first evaluation) and are not required to reproduce the reported tables.
Results are written to `results/`, tables and figures to `revision/`.

## Notes on the protocol

- Each class of both datasets is stored as one contiguous capture. The split is
  chronological within each class (70/10/20); preprocessing, the graph and the
  node prototypes use the training partition only.
- The primary flow aggregates are computed over the complete flow record. A
  causal variant (only packets up to the current one, flow key without the
  class label) is available through `exp_common.load_base("causal")`.
- "Parameter reuse across graph sizes" applies a trained model to another node
  partition of the same capture after the adjacency, descriptors, neighbor
  lists and prototypes are recomputed from unlabeled events of that partition.

## License

MIT (see `LICENSE`).
