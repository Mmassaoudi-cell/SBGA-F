# SBGA-F: Sparse Bidirectional Graph Attention with Focal Loss

Model code for the paper

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

This repository contains the model code only. It does not include datasets,
experiment scripts, result tables, figures, or the manuscript.

## Files

```
exp_common.py            SBGA-F network (SBGAFv2, masked sparse attention), graph
                         context (adjacency, prototypes, descriptors, top-k lists),
                         window construction, training loop, stacking meta-learner
models.py                Temporal encoders and benchmark architectures
                         (FNN, RNN, CNN, Transformer, GNN, TGNN, BiRAGNN)
candidate_models.py      Class-balanced focal loss, TCN encoder, aggregation modules
ft_transformer.py        FT-Transformer benchmark
common.py                Windowing and graph utilities (structural descriptors, top-k)
FINAL_MODEL_CONFIG.yaml  Frozen SBGA-F architecture and hyperparameters
requirements.txt         Python environment
```

## Environment

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended. On some PyTorch/cuDNN builds the float32
cuDNN LSTM kernel is slow; setting the environment variable
`SBGAF_NATIVE_LSTM=1` selects PyTorch's native LSTM kernel (same model, same
precision).

## Minimal example

```python
import numpy as np, torch
import exp_common as E

F, C, N = 54, 15, 100                      # features, classes, graph nodes
model = E.SBGAFv2(F, 4, C).to(E.DEVICE)    # hidden 192, 2 LSTM layers (defaults)

# graph context: column-normalized adjacency A[v, u] = P(u -> v) and node prototypes
A = np.random.rand(N, N).astype("float32"); A[A < 0.9] = 0
A = A / np.maximum(A.sum(0, keepdims=True), 1e-9)
ctx = E.GraphCtx(A, np.random.randn(N, F).astype("float32"), k=6)

x = torch.randn(32, 10, F, device=E.DEVICE)          # windows of 10 events
node = torch.randint(0, N, (32,), device=E.DEVICE)   # emitting node of each sample
logits = model(x, ctx.struct[node],
               ctx.proto[ctx.in_idx[node]], ctx.in_w[node], ctx.in_m[node],
               ctx.proto[ctx.out_idx[node]], ctx.out_w[node], ctx.out_m[node])

# stacking: logistic-regression meta-learner on validation probabilities
# meta = E.fit_meta([p_sbgaf_val, p_catboost_val], y_val)
```

`E.build_dataset` builds windows, the event-transition graph and the graph
context from a prepared feature table, and `E.train_sbgaf` trains the model
with the frozen hyperparameters (`E.HP`).

## Data

The paper uses the public Edge-IIoTset (Ferrag et al., 2022) and TON_IoT
(Moustafa, 2021) datasets, which are available from their authors.

## License

MIT (see `LICENSE`).
