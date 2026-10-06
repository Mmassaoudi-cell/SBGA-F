"""Same feature-noise / missing-feature stress tests as 10_robustness_generalization.py,
run against BiRAGNN (source method) and XGBoost (strongest benchmark) for comparison,
so the robustness curves aren't reported in isolation."""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
import importlib
scr = importlib.import_module("04_candidate_screening")
scr5 = importlib.import_module("05_classical_benchmarks")
from models import MODEL_SPECS, IDSModel
from candidate_models import class_weights_sqrt
import xgboost as xgb

DEVICE = scr.DEVICE
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 42


def to_t(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype)


def train_biragnn(data, seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    spec = MODEL_SPECS["BiRAGNN"]
    model = IDSModel(in_dim=data["in_dim"], hidden=128, layers=3, dropout=0.2, activation="relu",
                      temporal=spec["temporal"], graph=spec["graph"], n_classes=data["n_classes"]).to(DEVICE)
    tr, va = data["splits"]["train"], data["splits"]["val"]
    cw = to_t(class_weights_sqrt(tr["y"], data["n_classes"])).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr = to_t(tr["X_win"]); ytr = to_t(tr["y"], torch.long)
    Xva = to_t(va["X_win"]).to(DEVICE); yva = va["y"]
    nin_tr = to_t(tr["neigh_in"]); nout_tr = to_t(tr["neigh_out"])
    nin_va = to_t(va["neigh_in"]).to(DEVICE); nout_va = to_t(va["neigh_out"]).to(DEVICE)
    n = len(Xtr); best_f1, best_state, bad = -1, None, 0
    for epoch in range(25):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            logits = model(Xtr[idx].to(DEVICE), nin_tr[idx].to(DEVICE), nout_tr[idx].to(DEVICE))
            loss = loss_fn(logits, ytr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vpred = model(Xva, nin_va, nout_va).argmax(1).cpu().numpy()
        vf1 = f1_score(yva, vpred, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, best_state, bad = vf1, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8:
                break
    model.load_state_dict(best_state); model.eval()
    return model


def eval_biragnn(model, data, noise_std=0.0, missing_frac=0.0):
    s = data["splits"]["test"]
    X = s["X_win"].copy()
    if noise_std > 0:
        X = X + np.random.default_rng(0).normal(0, noise_std, size=X.shape).astype(np.float32)
    if missing_frac > 0:
        mask = np.random.default_rng(1).random(X.shape[-1]) < missing_frac
        X[..., mask] = 0.0
    Nin = to_t(s["neigh_in"]).to(DEVICE); Nout = to_t(s["neigh_out"]).to(DEVICE)
    with torch.no_grad():
        proba = F.softmax(model(to_t(X).to(DEVICE), Nin, Nout), dim=1).cpu().numpy()
    pred = proba.argmax(1)
    return scr.eval_metrics(s["y"], pred, proba, data["n_classes"])


def main():
    data = scr.prepare_data()
    model = train_biragnn(data)
    flat_data, n_classes, _ = scr5.prepare_flat_data()

    results = []
    for std in [0.0, 0.1, 0.25, 0.5, 1.0]:
        m = eval_biragnn(model, data, noise_std=std)
        m.update(model="BiRAGNN", experiment="feature_noise", noise_std=std)
        results.append(m)
        print(f"BiRAGNN noise_std={std}: macroF1={m['macro_F1']:.2f}", flush=True)
    for frac in [0.0, 0.1, 0.25, 0.5]:
        m = eval_biragnn(model, data, missing_frac=frac)
        m.update(model="BiRAGNN", experiment="missing_features", missing_frac=frac)
        results.append(m)
        print(f"BiRAGNN missing_frac={frac}: macroF1={m['macro_F1']:.2f}", flush=True)

    # XGBoost: retrain once, evaluate under noise/missing (flat features)
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, objective="multi:softprob",
                             num_class=n_classes, random_state=SEED, n_jobs=-1, eval_metric="mlogloss")
    clf.fit(flat_data["train"]["X"], flat_data["train"]["y"])
    Xte = flat_data["test"]["X"]; yte = flat_data["test"]["y"]
    for std in [0.0, 0.1, 0.25, 0.5, 1.0]:
        Xn = Xte + (np.random.default_rng(0).normal(0, std, size=Xte.shape).astype(np.float32) if std > 0 else 0)
        proba = clf.predict_proba(Xn); pred = proba.argmax(1)
        m = scr.eval_metrics(yte, pred, proba, n_classes)
        m.update(model="XGBoost", experiment="feature_noise", noise_std=std)
        results.append(m)
        print(f"XGBoost noise_std={std}: macroF1={m['macro_F1']:.2f}", flush=True)
    for frac in [0.0, 0.1, 0.25, 0.5]:
        Xm = Xte.copy()
        if frac > 0:
            mask = np.random.default_rng(1).random(Xm.shape[-1]) < frac
            Xm[:, mask] = 0.0
        proba = clf.predict_proba(Xm); pred = proba.argmax(1)
        m = scr.eval_metrics(yte, pred, proba, n_classes)
        m.update(model="XGBoost", experiment="missing_features", missing_frac=frac)
        results.append(m)
        print(f"XGBoost missing_frac={frac}: macroF1={m['macro_F1']:.2f}", flush=True)

    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "robustness_baseline_comparison.csv"), index=False)
    print("DONE")


if __name__ == "__main__":
    main()
