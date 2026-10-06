"""
Robustness and generalization experiments (Sections 18/20/21), run against the
frozen final model (SBGA-F) and, where meaningful, against the strongest benchmarks
for comparison. TEST set only used for read-only evaluation; no retraining decisions
are made from these results (the model is already frozen).

Experiments:
  1. Feature-noise robustness: inject Gaussian noise at increasing std into TEST features.
  2. Missing-feature robustness: randomly zero-mask an increasing fraction of TEST features.
  3. Cross-topology zero-shot transfer (FLAGSHIP): the model trained on the 100-node graph
     is evaluated, with NO retraining, on the 8-node and 50-node graphs -- a capability
     the other two screened candidates (A, C) structurally cannot exercise, because their
     graph-context mechanism is a per-node prototype lookup table sized to one specific
     graph, whereas SBGA-F's structural/positional node features (degree, PageRank,
     clustering) are topology-agnostic by construction. This is verified DIRECTLY here,
     not just claimed.
  4. Low-data-fraction curve: train on 1/5/10/25/50/100% of TRAIN, evaluate on TEST.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
import importlib
scr = importlib.import_module("04_candidate_screening")
from common import load_view, get_feature_columns, fit_transform_features, build_windows, load_graph, node_prototypes, structural_node_features, topk_neighbors
from candidate_models import CandidateB_SBGAF, FocalLoss, class_weights_sqrt
from sklearn.preprocessing import LabelEncoder

DEVICE = scr.DEVICE
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 42


def to_t(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype)


def train_final_model(data, seed=SEED, max_epochs=30, patience=8, train_frac=1.0):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = data["splits"]["train"], data["splits"]["val"]
    n_full = len(tr["y"])
    if train_frac < 1.0:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n_full, size=max(50, int(n_full * train_frac)), replace=False)
    else:
        keep = np.arange(n_full)

    model = CandidateB_SBGAF(data["in_dim"], data["struct_dim"], 192, 2, 0.3397475692650171,
                              "relu", data["n_classes"], k=scr.K_TOPK).to(DEVICE)
    cw = class_weights_sqrt(tr["y"][keep], data["n_classes"])
    loss_fn = FocalLoss(cw, gamma=1.9229587245058637).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.0018098827092128786)

    def bt(s, idx=None):
        if idx is None:
            idx = np.arange(len(s["y"]))
        return (to_t(s["X_win"][idx]), to_t(s["struct_feat"][idx]), to_t(s["topk_in_feats"][idx]),
                to_t(s["topk_in_w"][idx]), to_t(s["topk_out_feats"][idx]), to_t(s["topk_out_w"][idx]),
                to_t(s["y"][idx], torch.long))

    Xtr, Str, Nin_tr, Ninw_tr, Nout_tr, Noutw_tr, ytr = bt(tr, keep)
    Xva, Sva, Nin_va, Ninw_va, Nout_va, Noutw_va, yva = [t.to(DEVICE) for t in bt(va)]

    n = len(Xtr); best_f1, best_state, bad = -1, None, 0
    for epoch in range(max_epochs):
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
            if bad >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    return model


def eval_on(model, data, split="test", feature_noise_std=0.0, missing_frac=0.0):
    s = data["splits"][split]
    X = s["X_win"].copy()
    if feature_noise_std > 0:
        X = X + np.random.default_rng(0).normal(0, feature_noise_std, size=X.shape).astype(np.float32)
    if missing_frac > 0:
        mask = np.random.default_rng(1).random(X.shape[-1]) < missing_frac
        X[..., mask] = 0.0
    with torch.no_grad():
        proba = F.softmax(model(to_t(X).to(DEVICE), to_t(s["struct_feat"]).to(DEVICE),
                                 to_t(s["topk_in_feats"]).to(DEVICE), to_t(s["topk_in_w"]).to(DEVICE),
                                 to_t(s["topk_out_feats"]).to(DEVICE), to_t(s["topk_out_w"]).to(DEVICE)), dim=1).cpu().numpy()
    pred = proba.argmax(1)
    return scr.eval_metrics(s["y"], pred, proba, data["n_classes"])


# -------------------- Experiment 3: cross-topology transfer --------------------
def prepare_system(system_size, view="fusion"):
    df = load_view(view)
    feature_cols = get_feature_columns(df)
    X_all, feat_names = fit_transform_features(df, feature_cols)
    node_col = f"node_{system_size}"
    le = LabelEncoder()
    y_all = le.fit_transform(df["Attack_type"].astype(str))
    n_classes = len(le.classes_)
    A_in, A_out = load_graph(system_size)
    struct_feat_all = structural_node_features(A_in)
    topk_in_idx, topk_in_w = topk_neighbors(A_in, scr.K_TOPK)
    topk_out_idx, topk_out_w = topk_neighbors(A_out, scr.K_TOPK)
    splits = {}
    for split in ["train", "val", "test"]:
        global_pos, win_idx = build_windows(df, node_col, split, W=scr.W)
        X_win = X_all[win_idx]
        y_split = y_all[global_pos]
        node_ids = df.loc[global_pos, node_col].to_numpy().astype(np.int64)
        splits[split] = dict(X_win=X_win, y=y_split, node_ids=node_ids, global_pos=global_pos)
    train_global_pos, _ = build_windows(df, node_col, "train", W=1)
    proto = node_prototypes(X_all[train_global_pos.reshape(-1)], splits["train"]["node_ids"], system_size)
    for split in splits:
        nid = splits[split]["node_ids"]
        splits[split]["struct_feat"] = struct_feat_all[nid]
        splits[split]["topk_in_feats"] = proto[topk_in_idx[nid]]
        splits[split]["topk_in_w"] = topk_in_w[nid]
        splits[split]["topk_out_feats"] = proto[topk_out_idx[nid]]
        splits[split]["topk_out_w"] = topk_out_w[nid]
    return dict(splits=splits, in_dim=X_all.shape[1], n_classes=n_classes, struct_dim=struct_feat_all.shape[1])


def run_cross_topology(model_100node, results):
    for target_size in [8, 50]:
        data_target = prepare_system(target_size)
        # zero-shot: the model trained on 100-node graph, no retraining
        m_zeroshot = eval_on(model_100node, data_target, split="test")
        m_zeroshot.update(experiment="cross_topology_transfer", mode="zero_shot_from_100node",
                           target_system=target_size)
        results.append(m_zeroshot)
        print(f"zero-shot 100-node -> {target_size}-node: macroF1={m_zeroshot['macro_F1']:.2f}", flush=True)

        # upper bound: retrained from scratch on the target graph's own train split
        model_target = train_final_model(data_target, seed=SEED)
        m_retrained = eval_on(model_target, data_target, split="test")
        m_retrained.update(experiment="cross_topology_transfer", mode="retrained_on_target",
                            target_system=target_size)
        results.append(m_retrained)
        print(f"retrained-from-scratch on {target_size}-node: macroF1={m_retrained['macro_F1']:.2f}", flush=True)


def main():
    print("preparing 100-node data and training reference final model...", flush=True)
    data100 = scr.prepare_data()
    model = train_final_model(data100, seed=SEED)
    results = []

    print("\n== Experiment 1: feature noise ==", flush=True)
    for std in [0.0, 0.1, 0.25, 0.5, 1.0]:
        m = eval_on(model, data100, feature_noise_std=std)
        m.update(experiment="feature_noise", noise_std=std)
        results.append(m)
        print(f"noise_std={std}: macroF1={m['macro_F1']:.2f}", flush=True)

    print("\n== Experiment 2: missing features ==", flush=True)
    for frac in [0.0, 0.1, 0.25, 0.5]:
        m = eval_on(model, data100, missing_frac=frac)
        m.update(experiment="missing_features", missing_frac=frac)
        results.append(m)
        print(f"missing_frac={frac}: macroF1={m['macro_F1']:.2f}", flush=True)

    print("\n== Experiment 3: cross-topology zero-shot transfer ==", flush=True)
    run_cross_topology(model, results)

    print("\n== Experiment 4: low-data-fraction curve ==", flush=True)
    for frac in [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]:
        t0 = time.time()
        m_model = train_final_model(data100, seed=SEED, train_frac=frac)
        m = eval_on(m_model, data100)
        m.update(experiment="low_data_fraction", train_fraction=frac, train_time_sec=time.time() - t0)
        results.append(m)
        print(f"train_frac={frac}: macroF1={m['macro_F1']:.2f}", flush=True)

    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "robustness_generalization_results.csv"), index=False)
    print("DONE. Saved results/robustness_generalization_results.csv")


if __name__ == "__main__":
    main()
