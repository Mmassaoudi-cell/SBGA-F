"""Does SBGA-F-Stack (SBGA-F + CatBoost + LR meta-learner) preserve zero-shot
cross-topology transfer? CatBoost operates on raw per-sample features only (no node
identity), and the meta-learner is a linear combiner over two probability vectors --
neither depends on graph size N, so in principle the stack should transfer too. Tested
directly here using the seed-42 base models (already trained, retrained fresh for
target-graph inference consistency)."""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(__file__))
import importlib
scr = importlib.import_module("04_candidate_screening")
scr5 = importlib.import_module("05_classical_benchmarks")
rob = importlib.import_module("10_robustness_generalization")
stk = importlib.import_module("13_stacked_model")
from common import load_view, get_feature_columns, fit_transform_features, build_windows

DEVICE = scr.DEVICE
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 42


def flat_features_for_system(system_size, view="fusion"):
    df = load_view(view)
    feat_cols = get_feature_columns(df)
    X_all, _ = fit_transform_features(df, feat_cols)
    node_col = f"node_{system_size}"
    out = {}
    for split in ["train", "test"]:
        gp, _ = build_windows(df, node_col, split, W=1)
        out[split] = X_all[gp.reshape(-1)]
    return out


def main():
    print("training base models on 100-node data (seed 42)...", flush=True)
    data_100 = scr.prepare_data()
    flat_100 = flat_features_for_system(100)
    flat_data, n_classes, _ = scr5.prepare_flat_data()

    sbgaf, _ = stk.train_sbgaf(data_100, SEED)
    cat = CatBoostClassifier(iterations=300, depth=8, learning_rate=0.1, loss_function="MultiClass",
                              auto_class_weights="SqrtBalanced", random_seed=SEED, verbose=False)
    cat.fit(flat_data["train"]["X"], flat_data["train"]["y"])

    p_sbgaf_va = stk.sbgaf_proba(sbgaf, data_100, "val")
    p_cat_va = cat.predict_proba(flat_data["val"]["X"])
    y_va = data_100["splits"]["val"]["y"]
    meta = LogisticRegression(max_iter=2000)
    meta.fit(np.concatenate([p_sbgaf_va, p_cat_va], axis=1), y_va)

    results = []
    # same-topology sanity check
    p_sbgaf_te = stk.sbgaf_proba(sbgaf, data_100, "test")
    p_cat_te = cat.predict_proba(flat_data["test"]["X"])
    y_te = data_100["splits"]["test"]["y"]
    stack_proba = meta.predict_proba(np.concatenate([p_sbgaf_te, p_cat_te], axis=1))
    m = scr.eval_metrics(y_te, stack_proba.argmax(1), stack_proba, n_classes)
    m.update(model="SBGA-F-Stack", target_system=100, mode="same_topology")
    results.append(m)
    print(f"same-topology (100-node): macroF1={m['macro_F1']:.2f}", flush=True)

    for target_size in [8, 50]:
        data_t = rob.prepare_system(target_size)
        flat_t = flat_features_for_system(target_size)
        p_sbgaf_t = stk.sbgaf_proba(sbgaf, data_t, "test")   # zero-shot: no retraining
        p_cat_t = cat.predict_proba(flat_t["test"])            # zero-shot: no retraining
        y_t = data_t["splits"]["test"]["y"]
        stack_proba_t = meta.predict_proba(np.concatenate([p_sbgaf_t, p_cat_t], axis=1))
        m_t = scr.eval_metrics(y_t, stack_proba_t.argmax(1), stack_proba_t, data_t["n_classes"])
        m_t.update(model="SBGA-F-Stack", target_system=target_size, mode="zero_shot_transfer")
        results.append(m_t)
        print(f"zero-shot -> {target_size}-node: macroF1={m_t['macro_F1']:.2f}", flush=True)

        # also record standalone CatBoost zero-shot (no retrain) for comparison
        m_cat_t = scr.eval_metrics(y_t, p_cat_t.argmax(1), p_cat_t, data_t["n_classes"])
        m_cat_t.update(model="CatBoost_only", target_system=target_size, mode="zero_shot_transfer")
        results.append(m_cat_t)
        print(f"  (CatBoost-only zero-shot -> {target_size}-node: macroF1={m_cat_t['macro_F1']:.2f})", flush=True)

    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "stack_transfer_check.csv"), index=False)
    print("DONE. Saved results/stack_transfer_check.csv")


if __name__ == "__main__":
    main()
