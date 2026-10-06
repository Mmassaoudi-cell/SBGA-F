"""
Final, untouched-TEST-set evaluation (Section 15): the frozen final model (B_SBGAF,
per FINAL_MODEL_CONFIG.yaml) vs. 15 benchmarks spanning classical / boosting / deep /
graph categories (Section 14), each with 5 seeds, on the 100-node/fusion multiclass
(Attack_type, 15-class) config. TEST split is touched here for the first time in the
whole study.

Benchmarks (15) + final model (1) = 16 total:
  classical:  LogisticRegression, LinearSVM(SGD), kNN
  boosting:   RandomForest, XGBoost, LightGBM, CatBoost
  deep:       FNN, RNN(LSTM), CNN, Transformer, FT-Transformer
  graph:      GNN, TGNN, BiRAGNN (= SOURCE_METHOD_REPRODUCTION, mandatory benchmark)

Results -> results/final_test_results.csv (one row per model x seed)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
import importlib
scr = importlib.import_module("04_candidate_screening")
scr5 = importlib.import_module("05_classical_benchmarks")
scr6 = importlib.import_module("06_deep_benchmarks_multiclass")
from candidate_models import CandidateB_SBGAF, FocalLoss, class_weights_sqrt
from models import MODEL_SPECS, IDSModel
from ft_transformer import FTTransformer

DEVICE = scr.DEVICE
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
eval_metrics = scr.eval_metrics
SEEDS = [42, 123, 2024, 7, 99]


def to_t(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype)


# ------------------------- final model -------------------------
def run_final_model(data, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = data["splits"]["train"], data["splits"]["val"], data["splits"]["test"]
    model = CandidateB_SBGAF(data["in_dim"], data["struct_dim"], 192, 2, 0.3397475692650171,
                              "relu", data["n_classes"], k=scr.K_TOPK).to(DEVICE)
    cw = class_weights_sqrt(tr["y"], data["n_classes"])
    loss_fn = FocalLoss(cw, gamma=1.9229587245058637).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.0018098827092128786)

    def bt(s):
        return (to_t(s["X_win"]), to_t(s["struct_feat"]), to_t(s["topk_in_feats"]), to_t(s["topk_in_w"]),
                to_t(s["topk_out_feats"]), to_t(s["topk_out_w"]), to_t(s["y"], torch.long))

    Xtr, Str, Nin_tr, Ninw_tr, Nout_tr, Noutw_tr, ytr = bt(tr)
    Xva, Sva, Nin_va, Ninw_va, Nout_va, Noutw_va, yva = [t.to(DEVICE) for t in bt(va)]
    Xte, Ste, Nin_te, Ninw_te, Nout_te, Noutw_te, yte = [t.to(DEVICE) for t in bt(te)]

    n = len(Xtr); best_f1, best_state, bad = -1, None, 0
    t0 = time.time()
    for epoch in range(30):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            logits = model(Xtr[idx].to(DEVICE), Str[idx].to(DEVICE), Nin_tr[idx].to(DEVICE),
                            Ninw_tr[idx].to(DEVICE), Nout_tr[idx].to(DEVICE), Noutw_tr[idx].to(DEVICE))
            loss = loss_fn(logits, ytr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vpred = model(Xva, Sva, Nin_va, Ninw_va, Nout_va, Noutw_va).argmax(1).cpu().numpy()
        vf1 = f1_score(yva.cpu().numpy(), vpred, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, best_state, bad = vf1, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state); model.eval()
    t1 = time.time()
    with torch.no_grad():
        tlogits = model(Xte, Ste, Nin_te, Ninw_te, Nout_te, Noutw_te)
        tproba = F.softmax(tlogits, dim=1).cpu().numpy()
    infer_ms = (time.time() - t1) / len(yte) * 1000
    tpred = tproba.argmax(1)
    n_params = sum(p.numel() for p in model.parameters())
    m = eval_metrics(yte.cpu().numpy(), tpred, tproba, data["n_classes"])
    m.update(model="Proposed_SBGA-F", seed=seed, category="proposed", train_time_sec=train_time,
              infer_time_ms_per_sample=infer_ms, n_params=n_params)
    return m, (yte.cpu().numpy(), tpred, tproba)


# ------------------------- classical/boosting on TEST -------------------------
def run_classical_test(flat_data, n_classes, seed):
    tr, te = flat_data["train"], flat_data["test"]
    factories = {
        "LogisticRegression": (lambda s: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=s, n_jobs=-1), "classical"),
        "LinearSVM_SGD": (lambda s: SGDClassifier(loss="hinge", class_weight="balanced", random_state=s, max_iter=50), "classical"),
        "kNN": (lambda s: KNeighborsClassifier(n_neighbors=15, n_jobs=-1), "classical"),
        "RandomForest": (lambda s: RandomForestClassifier(n_estimators=200, max_depth=20, class_weight="balanced", random_state=s, n_jobs=-1), "boosting"),
        "XGBoost": (lambda s: xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, objective="multi:softprob", num_class=n_classes, random_state=s, n_jobs=-1, eval_metric="mlogloss"), "boosting"),
        "LightGBM": (lambda s: lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05, objective="multiclass", num_class=n_classes, class_weight="balanced", random_state=s, verbosity=-1), "boosting"),
        "CatBoost": (lambda s: CatBoostClassifier(iterations=300, depth=8, learning_rate=0.1, loss_function="MultiClass", auto_class_weights="SqrtBalanced", random_seed=s, verbose=False), "boosting"),
    }
    results = []
    for name, (factory, cat) in factories.items():
        t0 = time.time()
        clf = factory(seed)
        if name == "kNN":
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(tr["y"]), size=min(30000, len(tr["y"])), replace=False)
            clf.fit(tr["X"][idx], tr["y"][idx])
        else:
            clf.fit(tr["X"], tr["y"])
        train_time = time.time() - t0
        t1 = time.time()
        if name == "LinearSVM_SGD":
            scores = clf.decision_function(te["X"])
            if scores.ndim == 1:
                scores = np.stack([-scores, scores], axis=1)
            proba = np.exp(scores - scores.max(1, keepdims=True))
            proba = proba / proba.sum(1, keepdims=True)
            pred = clf.predict(te["X"])
        else:
            proba = clf.predict_proba(te["X"])
            pred = proba.argmax(1)
        infer_ms = (time.time() - t1) / len(te["y"]) * 1000
        m = eval_metrics(te["y"], pred, proba, n_classes)
        n_params = getattr(clf, "n_features_in_", 0)
        m.update(model=name, seed=seed, category=cat, train_time_sec=train_time,
                  infer_time_ms_per_sample=infer_ms, n_params=n_params)
        results.append(m)
        print(f"   {name} seed={seed}: macroF1={m['macro_F1']:.2f}", flush=True)
    return results


# ------------------------- deep / graph benchmarks on TEST -------------------------
def run_ids_model_test(model_name, data, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    spec = MODEL_SPECS[model_name]
    model = IDSModel(in_dim=data["in_dim"], hidden=128, layers=3, dropout=0.2, activation="relu",
                      temporal=spec["temporal"], graph=spec["graph"], n_classes=data["n_classes"]).to(DEVICE)
    tr, va, te = data["splits"]["train"], data["splits"]["val"], data["splits"]["test"]
    cw = to_t(class_weights_sqrt(tr["y"], data["n_classes"])).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr = to_t(tr["X_win"]); ytr = to_t(tr["y"], torch.long)
    Xva = to_t(va["X_win"]).to(DEVICE); yva = va["y"]
    Xte = to_t(te["X_win"]).to(DEVICE); yte = te["y"]
    graph_mode = spec["graph"]
    if graph_mode != "none":
        nin_tr = to_t(tr["neigh_in"]); nout_tr = to_t(tr["neigh_out"])
        nin_va = to_t(va["neigh_in"]).to(DEVICE); nout_va = to_t(va["neigh_out"]).to(DEVICE)
        nin_te = to_t(te["neigh_in"]).to(DEVICE); nout_te = to_t(te["neigh_out"]).to(DEVICE)
    n = len(Xtr); best_f1, best_state, bad = -1, None, 0
    t0 = time.time()
    for epoch in range(25):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            xb = Xtr[idx].to(DEVICE)
            logits = model(xb, nin_tr[idx].to(DEVICE), nout_tr[idx].to(DEVICE) if graph_mode == "bi" else None) if graph_mode != "none" else model(xb)
            loss = loss_fn(logits, ytr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vlogits = model(Xva, nin_va, nout_va if graph_mode == "bi" else None) if graph_mode != "none" else model(Xva)
            vpred = vlogits.argmax(1).cpu().numpy()
        vf1 = f1_score(yva, vpred, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, best_state, bad = vf1, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state); model.eval()
    t1 = time.time()
    with torch.no_grad():
        tlogits = model(Xte, nin_te, nout_te if graph_mode == "bi" else None) if graph_mode != "none" else model(Xte)
        tproba = F.softmax(tlogits, dim=1).cpu().numpy()
    infer_ms = (time.time() - t1) / len(yte) * 1000
    tpred = tproba.argmax(1)
    n_params = sum(p.numel() for p in model.parameters())
    cat = "graph" if graph_mode != "none" else "deep"
    if model_name == "BiRAGNN":
        cat = "source_method"
    m = eval_metrics(yte, tpred, tproba, data["n_classes"])
    m.update(model=model_name, seed=seed, category=cat, train_time_sec=train_time,
              infer_time_ms_per_sample=infer_ms, n_params=n_params)
    return m


def run_ft_transformer_test(data, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = data["splits"]["train"], data["splits"]["val"], data["splits"]["test"]
    Xtr_flat = tr["X_win"][:, -1, :]; Xva_flat = va["X_win"][:, -1, :]; Xte_flat = te["X_win"][:, -1, :]
    model = FTTransformer(n_features=data["in_dim"], d_token=32, n_layers=3, n_heads=4,
                           dropout=0.1, n_classes=data["n_classes"]).to(DEVICE)
    cw = to_t(class_weights_sqrt(tr["y"], data["n_classes"])).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr = to_t(Xtr_flat); ytr = to_t(tr["y"], torch.long)
    Xva = to_t(Xva_flat).to(DEVICE); yva = va["y"]
    Xte = to_t(Xte_flat).to(DEVICE); yte = te["y"]
    n = len(Xtr); best_f1, best_state, bad = -1, None, 0
    t0 = time.time()
    for epoch in range(25):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
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
            if bad >= 8:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state); model.eval()
    t1 = time.time()
    with torch.no_grad():
        tproba = F.softmax(model(Xte), dim=1).cpu().numpy()
    infer_ms = (time.time() - t1) / len(yte) * 1000
    tpred = tproba.argmax(1)
    n_params = sum(p.numel() for p in model.parameters())
    m = eval_metrics(yte, tpred, tproba, data["n_classes"])
    m.update(model="FT-Transformer", seed=seed, category="deep", train_time_sec=train_time,
              infer_time_ms_per_sample=infer_ms, n_params=n_params)
    return m


def main():
    print("preparing data (candidate-style, for final model + deep/graph benchmarks)...", flush=True)
    data = scr.prepare_data()
    print("preparing flat data (for classical/boosting benchmarks)...", flush=True)
    flat_data, n_classes, class_names = scr5.prepare_flat_data()

    all_results = []
    conf_store = {}

    for seed in SEEDS:
        print(f"\n===== SEED {seed} =====", flush=True)
        print("-- final model (Proposed_SBGA-F)", flush=True)
        m, conf = run_final_model(data, seed)
        print(f"   macroF1={m['macro_F1']:.2f} minRecall={m['min_class_recall']:.2f}", flush=True)
        all_results.append(m)
        conf_store[("Proposed_SBGA-F", seed)] = conf

        print("-- classical/boosting", flush=True)
        all_results.extend(run_classical_test(flat_data, n_classes, seed))

        for name in ["FNN", "RNN", "CNN", "Transformer", "GNN", "TGNN", "BiRAGNN"]:
            print(f"-- {name}", flush=True)
            m = run_ids_model_test(name, data, seed)
            print(f"   macroF1={m['macro_F1']:.2f} minRecall={m['min_class_recall']:.2f}", flush=True)
            all_results.append(m)

        print("-- FT-Transformer", flush=True)
        m = run_ft_transformer_test(data, seed)
        print(f"   macroF1={m['macro_F1']:.2f} minRecall={m['min_class_recall']:.2f}", flush=True)
        all_results.append(m)

        pd.DataFrame(all_results).to_csv(os.path.join(RESULTS_DIR, "final_test_results.csv"), index=False)

    # save one confusion-matrix-ready prediction set (seed 42) for the manuscript figure
    y_true, y_pred, y_proba = conf_store[("Proposed_SBGA-F", 42)]
    np.savez(os.path.join(RESULTS_DIR, "final_model_test_predictions_seed42.npz"),
             y_true=y_true, y_pred=y_pred, y_proba=y_proba, class_names=np.array(class_names, dtype=object))

    print("\nDONE. Saved results/final_test_results.csv")


if __name__ == "__main__":
    main()
