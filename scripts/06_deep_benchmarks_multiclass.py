"""
Multiclass retraining of the source paper's 7 architectures (FNN, RNN, CNN,
Transformer, GNN, TGNN, BiRAGNN) at the same 100-node/fusion config used for
candidate screening, so they're directly comparable to the proposed model on the
task-mandated primary metric (macro-F1 on genuine multiclass labels) -- the source
paper itself never does this (binary-only). Also trains FT-Transformer, a strong
modern tabular-Transformer competitor not in the source paper at all.

These, together with 05_classical_benchmarks.py and the frozen final candidate,
make up the >=10-benchmark final comparison (Section 14).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from common import load_view, get_feature_columns, fit_transform_features, build_windows, load_graph, node_prototypes
from models import IDSModel, MODEL_SPECS
from ft_transformer import FTTransformer
import importlib
scr = importlib.import_module("04_candidate_screening")
eval_metrics = scr.eval_metrics
from candidate_models import class_weights_sqrt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
W = 10
SYSTEM = 100
VIEW = "fusion"
HIDDEN = 128
LAYERS = 3
DROPOUT = 0.2
BATCH = 1024
MAX_EPOCHS = 25
PATIENCE = 8

MODELS_TO_RUN = ["FNN", "RNN", "CNN", "Transformer", "GNN", "TGNN", "BiRAGNN", "FT-Transformer"]


def to_t(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype)


def prepare():
    df = load_view(VIEW)
    feature_cols = get_feature_columns(df)
    X_all, feat_names = fit_transform_features(df, feature_cols)
    node_col = f"node_{SYSTEM}"
    le = LabelEncoder()
    y_all = le.fit_transform(df["Attack_type"].astype(str))
    n_classes = len(le.classes_)

    splits = {}
    for split in ["train", "val", "test"]:
        global_pos, win_idx = build_windows(df, node_col, split, W=W)
        X_win = X_all[win_idx]
        y_split = y_all[global_pos]
        node_ids = df.loc[global_pos, node_col].to_numpy().astype(np.int64)
        splits[split] = dict(X_win=X_win, y=y_split, node_ids=node_ids, global_pos=global_pos)

    A_in, A_out = load_graph(SYSTEM)
    train_global_pos, _ = build_windows(df, node_col, "train", W=1)
    proto = node_prototypes(X_all[train_global_pos.reshape(-1)], splits["train"]["node_ids"], SYSTEM)
    for split in splits:
        nid = splits[split]["node_ids"]
        splits[split]["neigh_in"] = A_in[nid] @ proto
        splits[split]["neigh_out"] = A_out[nid] @ proto

    return dict(splits=splits, in_dim=X_all.shape[1], n_classes=n_classes, class_names=le.classes_)


def run_ids_model(model_name, data, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    spec = MODEL_SPECS[model_name]
    model = IDSModel(in_dim=data["in_dim"], hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT,
                      activation="relu", temporal=spec["temporal"], graph=spec["graph"],
                      n_classes=data["n_classes"]).to(DEVICE)
    tr, va = data["splits"]["train"], data["splits"]["val"]
    cw = to_t(class_weights_sqrt(tr["y"], data["n_classes"])).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    Xtr = to_t(tr["X_win"]); ytr = to_t(tr["y"], torch.long)
    Xva = to_t(va["X_win"]).to(DEVICE); yva = va["y"]
    graph_mode = spec["graph"]
    if graph_mode != "none":
        nin_tr = to_t(tr["neigh_in"]); nout_tr = to_t(tr["neigh_out"])
        nin_va = to_t(va["neigh_in"]).to(DEVICE); nout_va = to_t(va["neigh_out"]).to(DEVICE)

    n = len(Xtr)
    best_f1, best_state, bad = -1, None, 0
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = Xtr[idx].to(DEVICE)
            if graph_mode != "none":
                logits = model(xb, nin_tr[idx].to(DEVICE), nout_tr[idx].to(DEVICE) if graph_mode == "bi" else None)
            else:
                logits = model(xb)
            loss = loss_fn(logits, ytr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            if graph_mode != "none":
                vlogits = model(Xva, nin_va, nout_va if graph_mode == "bi" else None)
            else:
                vlogits = model(Xva)
            vpred = vlogits.argmax(1).cpu().numpy()
        from sklearn.metrics import f1_score
        vf1 = f1_score(yva, vpred, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, best_state, bad = vf1, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        if graph_mode != "none":
            vlogits = model(Xva, nin_va, nout_va if graph_mode == "bi" else None)
        else:
            vlogits = model(Xva)
        vproba = F.softmax(vlogits, dim=1).cpu().numpy()
        vpred = vproba.argmax(1)
    n_params = sum(p.numel() for p in model.parameters())
    m = eval_metrics(yva, vpred, vproba, data["n_classes"])
    m.update(model=model_name, seed=seed, train_time_sec=train_time, n_params=n_params)
    return m


def run_ft_transformer(data, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = data["splits"]["train"], data["splits"]["val"]
    Xtr_flat = tr["X_win"][:, -1, :]; Xva_flat = va["X_win"][:, -1, :]
    model = FTTransformer(n_features=data["in_dim"], d_token=32, n_layers=3, n_heads=4,
                           dropout=0.1, n_classes=data["n_classes"]).to(DEVICE)
    cw = to_t(class_weights_sqrt(tr["y"], data["n_classes"])).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr = to_t(Xtr_flat); ytr = to_t(tr["y"], torch.long)
    Xva = to_t(Xva_flat).to(DEVICE); yva = va["y"]
    n = len(Xtr)
    best_f1, best_state, bad = -1, None, 0
    t0 = time.time()
    from sklearn.metrics import f1_score
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            logits = model(Xtr[idx].to(DEVICE))
            loss = loss_fn(logits, ytr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vpred = model(Xva).argmax(1).cpu().numpy()
        vf1 = f1_score(yva, vpred, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, best_state, bad = vf1, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        vproba = F.softmax(model(Xva), dim=1).cpu().numpy()
        vpred = vproba.argmax(1)
    n_params = sum(p.numel() for p in model.parameters())
    m = eval_metrics(yva, vpred, vproba, data["n_classes"])
    m.update(model="FT-Transformer", seed=seed, train_time_sec=train_time, n_params=n_params)
    return m


def main(seeds=(42, 123, 2024)):
    print("preparing data...", flush=True)
    data = prepare()
    results = []
    for name in MODELS_TO_RUN:
        for seed in seeds:
            print(f"-- {name} seed={seed}", flush=True)
            if name == "FT-Transformer":
                res = run_ft_transformer(data, seed)
            else:
                res = run_ids_model(name, data, seed)
            print(f"   macroF1={res['macro_F1']:.2f} minRecall={res['min_class_recall']:.2f} "
                  f"time={res['train_time_sec']:.1f}s", flush=True)
            results.append(res)
    out = pd.DataFrame(results)
    out.to_csv(os.path.join(RESULTS_DIR, "deep_benchmarks_multiclass_val.csv"), index=False)
    print("Saved results/deep_benchmarks_multiclass_val.csv")


if __name__ == "__main__":
    main()
