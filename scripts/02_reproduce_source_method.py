"""
SOURCE_METHOD_REPRODUCTION: trains and evaluates the source paper's full protocol
(FNN, RNN, CNN, Transformer, GNN, TGNN, BiRAGNN) across the paper's exact table
structure (standalone + 8/50/100-node systems x packet/flow/fusion views), on the
substituted Edge-IIoTset data (see DATA_AUDIT.md), using the paper's own reported
optimal hyperparameters (Tables I-II) wherever they apply.

Results -> results/reproduction_results.csv (one row per system x view x model),
seed-level detail -> results/reproduction_raw.jsonl
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, recall_score, f1_score, precision_score,
                              confusion_matrix, average_precision_score)

sys.path.insert(0, os.path.dirname(__file__))
from common import load_view, get_feature_columns, fit_transform_features, build_windows, load_graph, node_prototypes
from models import IDSModel, MODEL_SPECS, GRAPH_MODELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
W = 10
MAX_EPOCHS = 35
PATIENCE = 10
BATCH = 1024
SEED = 42

OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}

# Paper's optimal hyperparameters (Tables I & II). Missing dropout values (CNN,
# standalone) default to 0.2 (documented assumption, paper leaves them blank).
HP = {
    ("FNN", "standalone"): dict(L=3, U=64, D=0.2, O="adam", A="relu"),
    ("FNN", "nnode"):      dict(L=5, U=32, D=0.2, O="adam", A="relu"),
    ("RNN", "standalone"): dict(L=3, U=32, D=0.0, O="sgd",  A="tanh"),
    ("RNN", "nnode"):      dict(L=3, U=16, D=0.0, O="adam", A="relu"),
    ("CNN", "standalone"): dict(L=4, U=32, D=0.2, O="rmsprop", A="relu"),
    ("CNN", "nnode"):      dict(L=6, U=64, D=0.2, O="adam", A="relu"),
    ("Transformer", "standalone"): dict(L=3, U=128, D=0.2, O="adam", A="relu"),
    ("Transformer", "nnode"):      dict(L=12, U=256, D=0.1, O="adam", A="relu"),
    ("GNN", "nnode"):     dict(L=5, U=128, D=0.2, O="rmsprop", A="relu"),
    ("TGNN", "nnode"):    dict(L=4, U=64, D=0.2, O="adam", A="relu"),
    ("BiRAGNN", "nnode"): dict(L=5, U=128, D=0.2, O="adam", A="relu"),
}


def get_hp(model_name, system_size):
    key = "standalone" if system_size == 1 else "nnode"
    return HP[(model_name, key)]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_system_view(system_size, view):
    df = load_view(view)
    feature_cols = get_feature_columns(df)
    X_all, feat_names = fit_transform_features(df, feature_cols)
    node_col = f"node_{system_size}"
    y_all = df["Attack_label"].to_numpy().astype(np.float32)

    splits = {}
    for split in ["train", "val", "test"]:
        global_pos, win_idx = build_windows(df, node_col, split, W=W)
        X_win = X_all[win_idx]
        y_split = y_all[global_pos]
        node_ids = df.loc[global_pos, node_col].to_numpy().astype(np.int64)
        splits[split] = dict(X_win=X_win, y=y_split, node_ids=node_ids)

    if system_size > 1:
        A_in, A_out = load_graph(system_size)
        # node prototype = mean feature vector (own, non-windowed) of a node's TRAIN rows
        train_global_pos, _ = build_windows(df, node_col, "train", W=1)
        train_global_pos = train_global_pos.reshape(-1)
        proto = node_prototypes(X_all[train_global_pos], splits["train"]["node_ids"], system_size)
        for split in splits:
            nid = splits[split]["node_ids"]
            splits[split]["neigh_in"] = A_in[nid] @ proto
            splits[split]["neigh_out"] = A_out[nid] @ proto

    return dict(splits=splits, in_dim=X_all.shape[1], n_feat=len(feat_names))


def to_tensor(a):
    return torch.as_tensor(a, dtype=torch.float32)


def run_one(model_name, system_size, view, data, seed=SEED):
    set_seed(seed)
    spec = MODEL_SPECS[model_name]
    hp = get_hp(model_name, system_size)
    graph_mode = spec["graph"] if system_size > 1 else "none"

    model = IDSModel(in_dim=data["in_dim"], hidden=hp["U"], layers=hp["L"], dropout=hp["D"],
                      activation=hp["A"], temporal=spec["temporal"], graph=graph_mode, n_classes=1).to(DEVICE)
    # Paper gives no LR/momentum; SGD with a small LR and no momentum was found (in
    # diagnostic testing) to plateau at majority-class prediction for ~8 epochs before
    # breaking through -- an artifact of vanilla SGD's slow start, not a real property
    # of the method. We use the standard robust default lr=0.1/momentum=0.9 for SGD.
    if hp["O"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    else:
        optimizer = OPTIMIZERS[hp["O"]](model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    tr = data["splits"]["train"]; va = data["splits"]["val"]; te = data["splits"]["test"]
    Xtr = to_tensor(tr["X_win"]); ytr = to_tensor(tr["y"])
    Xva = to_tensor(va["X_win"]).to(DEVICE); yva = va["y"]
    Xte = to_tensor(te["X_win"]).to(DEVICE); yte = te["y"]

    if graph_mode != "none":
        nin_tr = to_tensor(tr["neigh_in"]); nout_tr = to_tensor(tr["neigh_out"])
        nin_va = to_tensor(va["neigh_in"]).to(DEVICE); nout_va = to_tensor(va["neigh_out"]).to(DEVICE)
        nin_te = to_tensor(te["neigh_in"]).to(DEVICE); nout_te = to_tensor(te["neigh_out"]).to(DEVICE)

    n = len(Xtr)
    best_val_f1, best_state, bad_epochs = -1, None, 0
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = Xtr[idx].to(DEVICE); yb = ytr[idx].to(DEVICE)
            if graph_mode != "none":
                logits = model(xb, nin_tr[idx].to(DEVICE), nout_tr[idx].to(DEVICE) if graph_mode == "bi" else None)
            else:
                logits = model(xb)
            loss = loss_fn(logits.squeeze(-1), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            if graph_mode != "none":
                vlogits = model(Xva, nin_va, nout_va if graph_mode == "bi" else None)
            else:
                vlogits = model(Xva)
            vpred = (torch.sigmoid(vlogits.squeeze(-1)) > 0.5).cpu().numpy().astype(int)
        vf1 = f1_score(yva.astype(int), vpred, average="macro", zero_division=0)
        if vf1 > best_val_f1:
            best_val_f1 = vf1; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        if graph_mode != "none":
            tlogits = model(Xte, nin_te, nout_te if graph_mode == "bi" else None)
        else:
            tlogits = model(Xte)
        tprob = torch.sigmoid(tlogits.squeeze(-1)).cpu().numpy()
    infer_time_ms = (time.time() - t1) / max(1, len(yte)) * 1000
    tpred = (tprob > 0.5).astype(int)
    yte_int = yte.astype(int)

    tn, fp, fn, tp = confusion_matrix(yte_int, tpred, labels=[0, 1]).ravel()
    ac = accuracy_score(yte_int, tpred)
    dr = recall_score(yte_int, tpred, pos_label=1, zero_division=0)
    fa = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    macro_f1 = f1_score(yte_int, tpred, average="macro", zero_division=0)
    weighted_f1 = f1_score(yte_int, tpred, average="weighted", zero_division=0)
    precision = precision_score(yte_int, tpred, pos_label=1, zero_division=0)
    pr_auc = average_precision_score(yte_int, tprob)
    n_params = sum(p.numel() for p in model.parameters())

    return dict(
        system_size=system_size, view=view, model=model_name, seed=seed,
        AC=ac * 100, DR=dr * 100, FA=fa * 100, macro_F1=macro_f1 * 100, weighted_F1=weighted_f1 * 100,
        precision=precision * 100, PR_AUC=pr_auc, best_val_macroF1=best_val_f1 * 100,
        train_time_sec=train_time, infer_time_ms_per_sample=infer_time_ms, n_params=n_params,
        n_train=len(ytr), n_test=len(yte),
    )


def main():
    all_results = []
    raw_path = os.path.join(RESULTS_DIR, "reproduction_raw.jsonl")
    with open(raw_path, "w") as rawf:
        for system_size in [1, 8, 50, 100]:
            for view in ["packet_level", "flow_level", "fusion"]:
                print(f"=== system={system_size} view={view}: preparing data ===", flush=True)
                data = prepare_system_view(system_size, view)
                models_here = ["FNN", "RNN", "CNN", "Transformer"]
                if system_size > 1:
                    models_here += ["GNN", "TGNN", "BiRAGNN"]
                for m in models_here:
                    print(f"  -> training {m} ...", flush=True)
                    res = run_one(m, system_size, view, data, seed=SEED)
                    print(f"     AC={res['AC']:.2f} DR={res['DR']:.2f} FA={res['FA']:.2f} macroF1={res['macro_F1']:.2f} "
                          f"(train {res['train_time_sec']:.1f}s)", flush=True)
                    all_results.append(res)
                    rawf.write(json.dumps(res) + "\n")
                    rawf.flush()

    import pandas as pd
    out = pd.DataFrame(all_results)
    out.to_csv(os.path.join(RESULTS_DIR, "reproduction_results.csv"), index=False)
    print("DONE. Saved results/reproduction_results.csv")


if __name__ == "__main__":
    main()
