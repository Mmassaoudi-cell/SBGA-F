"""
Stage 1 (smoke test) + Stage 2 (validation-only, multi-seed screening) for the 4
hybrid candidates in MODEL_CANDIDATES.md, per Section 9 of the task spec. TEST split
is never touched here. Screening config: 100-node system (richest graph), fusion view
(richest features) -- the paper's own headline configuration -- multiclass target
(Attack_type, 15 classes), matching the task's mandate that macro-F1 on genuine
multiclass labels is the primary metric.

Results -> results/candidate_screening.csv
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (f1_score, recall_score, precision_score, balanced_accuracy_score,
                              roc_auc_score)
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from common import (load_view, get_feature_columns, fit_transform_features, build_windows,
                     load_graph, node_prototypes, structural_node_features, topk_neighbors)
from candidate_models import (CandidateB_SBGAF, CandidateA_GraphBranch, CandidateC_TCNGXD,
                               CandidateD_EBGN, FocalLoss, class_weights_sqrt, distillation_loss,
                               evidential_loss)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
W = 10
K_TOPK = 6
SYSTEM = 100
VIEW = "fusion"
HIDDEN = 128
LAYERS = 3
DROPOUT = 0.2
BATCH = 1024


def to_t(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype)


def prepare_data():
    df = load_view(VIEW)
    feature_cols = get_feature_columns(df)
    X_all, feat_names = fit_transform_features(df, feature_cols)
    node_col = f"node_{SYSTEM}"

    le = LabelEncoder()
    y_all = le.fit_transform(df["Attack_type"].astype(str))
    n_classes = len(le.classes_)

    A_in, A_out = load_graph(SYSTEM)
    struct_feat_all = structural_node_features(A_in)  # [n_nodes,4]
    topk_in_idx, topk_in_w = topk_neighbors(A_in, K_TOPK)
    topk_out_idx, topk_out_w = topk_neighbors(A_out, K_TOPK)

    splits = {}
    for split in ["train", "val", "test"]:
        global_pos, win_idx = build_windows(df, node_col, split, W=W)
        X_win = X_all[win_idx]
        y_split = y_all[global_pos]
        node_ids = df.loc[global_pos, node_col].to_numpy().astype(np.int64)
        splits[split] = dict(X_win=X_win, y=y_split, node_ids=node_ids, global_pos=global_pos)

    train_global_pos, _ = build_windows(df, node_col, "train", W=1)
    train_flat_pos = train_global_pos.reshape(-1)
    proto = node_prototypes(X_all[train_flat_pos], splits["train"]["node_ids"], SYSTEM)

    for split in splits:
        nid = splits[split]["node_ids"]
        splits[split]["neigh_in"] = A_in[nid] @ proto
        splits[split]["neigh_out"] = A_out[nid] @ proto
        splits[split]["struct_feat"] = struct_feat_all[nid]
        splits[split]["topk_in_feats"] = proto[topk_in_idx[nid]]     # [n,k,F]
        splits[split]["topk_in_w"] = topk_in_w[nid]                   # [n,k]
        splits[split]["topk_out_feats"] = proto[topk_out_idx[nid]]
        splits[split]["topk_out_w"] = topk_out_w[nid]
        splits[split]["node_w_row_in"] = A_in[nid]                     # [n, n_nodes] for CrossAttnPool
        # flat (last-timestep) features, for the GBM teacher/gate
        splits[split]["X_flat"] = X_all[splits[split]["global_pos"]]

    return dict(splits=splits, in_dim=X_all.shape[1], n_classes=n_classes, n_nodes=SYSTEM,
                proto=proto, struct_dim=struct_feat_all.shape[1], class_names=le.classes_)


def eval_metrics(y_true, y_pred, y_proba, n_classes):
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=list(range(n_classes)))
    except Exception:
        auc = float("nan")
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(n_classes)))
    min_class_recall = float(np.min(per_class_recall))
    return dict(macro_F1=macro_f1 * 100, weighted_F1=weighted_f1 * 100, balanced_acc=bal_acc * 100,
                macro_recall=macro_recall * 100, macro_precision=macro_precision * 100,
                macro_auc_ovr=auc, min_class_recall=min_class_recall * 100)


def train_gbm_teacher(data, seed):
    tr = data["splits"]["train"]
    clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=63, learning_rate=0.05,
                              objective="multiclass", num_class=data["n_classes"],
                              class_weight="balanced", random_state=seed, verbosity=-1)
    clf.fit(tr["X_flat"], tr["y"])
    return clf


def run_candidate_B(data, seed, max_epochs=20, patience=6):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = data["splits"]["train"], data["splits"]["val"], data["splits"]["test"]
    model = CandidateB_SBGAF(data["in_dim"], data["struct_dim"], HIDDEN, LAYERS, DROPOUT, "relu",
                              data["n_classes"], k=K_TOPK).to(DEVICE)
    cw = class_weights_sqrt(tr["y"], data["n_classes"])
    loss_fn = FocalLoss(cw, gamma=2.0).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def batch_tensors(split):
        return (to_t(split["X_win"]), to_t(split["struct_feat"]), to_t(split["topk_in_feats"]),
                to_t(split["topk_in_w"]), to_t(split["topk_out_feats"]), to_t(split["topk_out_w"]),
                to_t(split["y"], torch.long))

    Xtr, Str, Nin_tr, Ninw_tr, Nout_tr, Noutw_tr, ytr = batch_tensors(tr)
    Xva, Sva, Nin_va, Ninw_va, Nout_va, Noutw_va, yva = [t.to(DEVICE) for t in batch_tensors(va)]
    Xte, Ste, Nin_te, Ninw_te, Nout_te, Noutw_te, yte = [t.to(DEVICE) for t in batch_tensors(te)]

    n = len(Xtr)
    best_f1, best_state, bad = -1, None, 0
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            logits = model(Xtr[idx].to(DEVICE), Str[idx].to(DEVICE), Nin_tr[idx].to(DEVICE),
                            Ninw_tr[idx].to(DEVICE), Nout_tr[idx].to(DEVICE), Noutw_tr[idx].to(DEVICE))
            loss = loss_fn(logits, ytr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vlogits = model(Xva, Sva, Nin_va, Ninw_va, Nout_va, Noutw_va)
            vpred = vlogits.argmax(1).cpu().numpy()
        vf1 = f1_score(yva.cpu().numpy(), vpred, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, best_state, bad = vf1, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        vlogits = model(Xva, Sva, Nin_va, Ninw_va, Nout_va, Noutw_va)
        vproba = F.softmax(vlogits, dim=1).cpu().numpy()
        vpred = vproba.argmax(1)
    n_params = sum(p.numel() for p in model.parameters())
    m = eval_metrics(yva.cpu().numpy(), vpred, vproba, data["n_classes"])
    m.update(candidate="B_SBGAF", seed=seed, train_time_sec=train_time, n_params=n_params)
    return m


def run_candidate_A(data, seed, max_epochs=15, patience=5):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = data["splits"]["train"], data["splits"]["val"]
    t0 = time.time()
    gbm = train_gbm_teacher(data, seed)
    gbm_time = time.time() - t0

    model = CandidateA_GraphBranch(data["in_dim"], HIDDEN // 2, 2, DROPOUT, "relu", data["n_classes"]).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr = to_t(tr["X_win"]); Nin_tr = to_t(tr["neigh_in"]); Nout_tr = to_t(tr["neigh_out"]); ytr = to_t(tr["y"], torch.long)
    Xva = to_t(va["X_win"]).to(DEVICE); Nin_va = to_t(va["neigh_in"]).to(DEVICE); Nout_va = to_t(va["neigh_out"]).to(DEVICE)
    yva = va["y"]

    n = len(Xtr)
    best_f1, best_state, bad = -1, None, 0
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            logits = model(Xtr[idx].to(DEVICE), Nin_tr[idx].to(DEVICE), Nout_tr[idx].to(DEVICE))
            loss = loss_fn(logits, ytr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vpred = model(Xva, Nin_va, Nout_va).argmax(1).cpu().numpy()
        vf1 = f1_score(yva, vpred, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, best_state, bad = vf1, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    train_time = time.time() - t0 + gbm_time
    model.load_state_dict(best_state); model.eval()

    # gate: route by GBM confidence, tuned threshold on validation macro-F1
    gbm_proba_va = gbm.predict_proba(va["X_flat"])
    gbm_conf_va = gbm_proba_va.max(1)
    gbm_pred_va = gbm_proba_va.argmax(1)
    with torch.no_grad():
        graph_proba_va = F.softmax(model(Xva, Nin_va, Nout_va), dim=1).cpu().numpy()
    graph_pred_va = graph_proba_va.argmax(1)

    best_tau, best_tau_f1 = 0.5, -1
    for tau in np.linspace(0.3, 0.95, 14):
        routed = gbm_conf_va < tau
        pred = np.where(routed, graph_pred_va, gbm_pred_va)
        f1 = f1_score(yva, pred, average="macro", zero_division=0)
        if f1 > best_tau_f1:
            best_tau_f1, best_tau = f1, tau

    routed = gbm_conf_va < best_tau
    final_pred = np.where(routed, graph_pred_va, gbm_pred_va)
    final_proba = np.where(routed[:, None], graph_proba_va, gbm_proba_va)
    n_params = sum(p.numel() for p in model.parameters())
    m = eval_metrics(yva, final_pred, final_proba, data["n_classes"])
    m.update(candidate="A_GTCascade", seed=seed, train_time_sec=train_time, n_params=n_params,
              routed_fraction=float(routed.mean()), tau=best_tau)
    return m


def run_candidate_C(data, seed, max_epochs=20, patience=6):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = data["splits"]["train"], data["splits"]["val"]
    t0 = time.time()
    gbm = train_gbm_teacher(data, seed)
    teacher_proba_tr = gbm.predict_proba(tr["X_flat"])
    gbm_time = time.time() - t0

    proto_all = to_t(data["proto"]).to(DEVICE)
    model = CandidateC_TCNGXD(data["in_dim"], HIDDEN, LAYERS, DROPOUT, "relu", data["n_classes"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    Xtr = to_t(tr["X_win"]); ytr = to_t(tr["y"], torch.long); Wtr = to_t(tr["node_w_row_in"])
    Ttr = to_t(teacher_proba_tr)
    Xva = to_t(va["X_win"]).to(DEVICE); Wva = to_t(va["node_w_row_in"]).to(DEVICE); yva = va["y"]

    n = len(Xtr)
    best_f1, best_state, bad = -1, None, 0
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            logits = model(Xtr[idx].to(DEVICE), proto_all, Wtr[idx].to(DEVICE))
            loss = distillation_loss(logits, Ttr[idx].to(DEVICE), ytr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vpred = model(Xva, proto_all, Wva).argmax(1).cpu().numpy()
        vf1 = f1_score(yva, vpred, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, best_state, bad = vf1, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    train_time = time.time() - t0 + gbm_time
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        vproba = F.softmax(model(Xva, proto_all, Wva), dim=1).cpu().numpy()
        vpred = vproba.argmax(1)
    n_params = sum(p.numel() for p in model.parameters())
    m = eval_metrics(yva, vpred, vproba, data["n_classes"])
    m.update(candidate="C_TCNGXD", seed=seed, train_time_sec=train_time, n_params=n_params)
    return m


def run_candidate_D(data, seed, max_epochs=15, patience=5):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = data["splits"]["train"], data["splits"]["val"]
    model = CandidateD_EBGN(data["in_dim"], HIDDEN // 2, 2, DROPOUT, "relu", data["n_classes"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr = to_t(tr["X_win"]); Nin_tr = to_t(tr["neigh_in"]); Nout_tr = to_t(tr["neigh_out"]); ytr = to_t(tr["y"], torch.long)
    Xva = to_t(va["X_win"]).to(DEVICE); Nin_va = to_t(va["neigh_in"]).to(DEVICE); Nout_va = to_t(va["neigh_out"]).to(DEVICE)
    yva = va["y"]
    n = len(Xtr)
    best_f1, best_state, bad = -1, None, 0
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            evidence = model(Xtr[idx].to(DEVICE), Nin_tr[idx].to(DEVICE), Nout_tr[idx].to(DEVICE))
            loss = evidential_loss(evidence, ytr[idx].to(DEVICE), data["n_classes"], epoch)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            ev = model(Xva, Nin_va, Nout_va)
            alpha = ev + 1
            vpred = (alpha / alpha.sum(1, keepdim=True)).argmax(1).cpu().numpy()
        vf1 = f1_score(yva, vpred, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, best_state, bad = vf1, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        ev = model(Xva, Nin_va, Nout_va)
        alpha = ev + 1
        vproba = (alpha / alpha.sum(1, keepdim=True)).cpu().numpy()
        vpred = vproba.argmax(1)
        uncertainty = (data["n_classes"] / alpha.sum(1)).mean().item()
    n_params = sum(p.numel() for p in model.parameters())
    m = eval_metrics(yva, vpred, vproba, data["n_classes"])
    m.update(candidate="D_EBGN", seed=seed, train_time_sec=train_time, n_params=n_params,
              mean_uncertainty=uncertainty)
    return m


def main():
    print("preparing multiclass 100-node fusion data...", flush=True)
    data = prepare_data()
    print(f"n_classes={data['n_classes']} classes={list(data['class_names'])}", flush=True)
    print(f"train n={len(data['splits']['train']['y'])} val n={len(data['splits']['val']['y'])}", flush=True)

    runners = {"A_GTCascade": run_candidate_A, "B_SBGAF": run_candidate_B,
               "C_TCNGXD": run_candidate_C, "D_EBGN": run_candidate_D}

    print("\n=== STAGE 1: smoke test (1 seed, reduced budget) ===", flush=True)
    smoke_results = []
    for name, fn in runners.items():
        print(f"-- smoke: {name}", flush=True)
        res = fn(data, seed=42, max_epochs=6, patience=3) if name != "A_GTCascade" else fn(data, seed=42, max_epochs=6, patience=3)
        print(f"   {res}", flush=True)
        smoke_results.append(res)
    pd.DataFrame(smoke_results).to_csv(os.path.join(RESULTS_DIR, "candidate_smoke_test.csv"), index=False)

    print("\n=== STAGE 2: validation screening (3 seeds, full budget), candidates A/B/C ===", flush=True)
    screening_results = []
    for name in ["A_GTCascade", "B_SBGAF", "C_TCNGXD"]:
        fn = runners[name]
        for seed in [42, 123, 2024]:
            print(f"-- screening: {name} seed={seed}", flush=True)
            res = fn(data, seed=seed)
            print(f"   macroF1={res['macro_F1']:.2f} minRecall={res['min_class_recall']:.2f} "
                  f"params={res['n_params']} time={res['train_time_sec']:.1f}s", flush=True)
            screening_results.append(res)
    df = pd.DataFrame(screening_results)
    df.to_csv(os.path.join(RESULTS_DIR, "candidate_screening.csv"), index=False)
    print("\nDONE. Saved results/candidate_screening.csv and candidate_smoke_test.csv")


if __name__ == "__main__":
    main()
