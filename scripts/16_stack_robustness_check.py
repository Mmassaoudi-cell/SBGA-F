"""Does SBGA-F-Stack inherit CatBoost's catastrophic feature-noise fragility, since the
meta-learner consumes CatBoost's predictions as an input? Tested directly."""
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
stk = importlib.import_module("13_stacked_model")

DEVICE = scr.DEVICE
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 42


def main():
    data = scr.prepare_data()
    flat_data, n_classes, _ = scr5.prepare_flat_data()

    sbgaf, _ = stk.train_sbgaf(data, SEED)
    cat = CatBoostClassifier(iterations=300, depth=8, learning_rate=0.1, loss_function="MultiClass",
                              auto_class_weights="SqrtBalanced", random_seed=SEED, verbose=False)
    cat.fit(flat_data["train"]["X"], flat_data["train"]["y"])

    p_sbgaf_va = stk.sbgaf_proba(sbgaf, data, "val")
    p_cat_va = cat.predict_proba(flat_data["val"]["X"])
    y_va = data["splits"]["val"]["y"]
    meta = LogisticRegression(max_iter=2000)
    meta.fit(np.concatenate([p_sbgaf_va, p_cat_va], axis=1), y_va)

    s_te = data["splits"]["test"]
    Xte_win = s_te["X_win"]
    Xte_flat = flat_data["test"]["X"]
    y_te = s_te["y"]

    results = []
    for std in [0.0, 0.1, 0.25, 0.5, 1.0]:
        rng = np.random.default_rng(0)
        Xn_win = Xte_win + (rng.normal(0, std, size=Xte_win.shape).astype(np.float32) if std > 0 else 0)
        Xn_flat = Xte_flat + (rng.normal(0, std, size=Xte_flat.shape).astype(np.float32) if std > 0 else 0)

        with torch.no_grad():
            p_sbgaf = F.softmax(sbgaf(torch.as_tensor(Xn_win, dtype=torch.float32).to(DEVICE),
                                       torch.as_tensor(s_te["struct_feat"], dtype=torch.float32).to(DEVICE),
                                       torch.as_tensor(s_te["topk_in_feats"], dtype=torch.float32).to(DEVICE),
                                       torch.as_tensor(s_te["topk_in_w"], dtype=torch.float32).to(DEVICE),
                                       torch.as_tensor(s_te["topk_out_feats"], dtype=torch.float32).to(DEVICE),
                                       torch.as_tensor(s_te["topk_out_w"], dtype=torch.float32).to(DEVICE)), dim=1).cpu().numpy()
        p_cat = cat.predict_proba(Xn_flat)
        stack_proba = meta.predict_proba(np.concatenate([p_sbgaf, p_cat], axis=1))

        m_stack = scr.eval_metrics(y_te, stack_proba.argmax(1), stack_proba, n_classes)
        m_stack.update(model="SBGA-F-Stack", noise_std=std)
        m_sbgaf = scr.eval_metrics(y_te, p_sbgaf.argmax(1), p_sbgaf, n_classes)
        m_sbgaf.update(model="SBGA-F_component", noise_std=std)
        m_cat = scr.eval_metrics(y_te, p_cat.argmax(1), p_cat, n_classes)
        m_cat.update(model="CatBoost_component", noise_std=std)
        results.extend([m_stack, m_sbgaf, m_cat])
        print(f"std={std}: Stack={m_stack['macro_F1']:.2f} SBGA-F={m_sbgaf['macro_F1']:.2f} CatBoost={m_cat['macro_F1']:.2f}", flush=True)

    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "stack_robustness_check.csv"), index=False)
    print("DONE")


if __name__ == "__main__":
    main()
