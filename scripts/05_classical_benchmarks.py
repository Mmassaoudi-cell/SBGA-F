"""
Classical/boosting benchmark suite (Section 14 of the task spec): Logistic Regression,
linear SVM, k-NN, Random Forest, XGBoost, LightGBM, CatBoost -- multiclass, on the
flat (non-windowed, non-graph) tabular features. Reused later as part of the >=10
final benchmark comparison against the proposed model.

Run at the same screening config (100-node, fusion, multiclass Attack_type) so results
are directly comparable to the candidate screening / final-model numbers.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(__file__))
from common import load_view, get_feature_columns, fit_transform_features, build_windows
from sklearn.preprocessing import LabelEncoder

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SYSTEM = 100
VIEW = "fusion"

import importlib
scr = importlib.import_module("04_candidate_screening")
eval_metrics = scr.eval_metrics


def prepare_flat_data():
    df = load_view(VIEW)
    feature_cols = get_feature_columns(df)
    X_all, feat_names = fit_transform_features(df, feature_cols)
    node_col = f"node_{SYSTEM}"
    le = LabelEncoder()
    y_all = le.fit_transform(df["Attack_type"].astype(str))
    n_classes = len(le.classes_)
    out = {}
    for split in ["train", "val", "test"]:
        global_pos, _ = build_windows(df, node_col, split, W=1)
        pos = global_pos.reshape(-1)
        out[split] = dict(X=X_all[pos], y=y_all[pos])
    return out, n_classes, le.classes_


def run_all(seeds=(42, 123, 2024)):
    data, n_classes, class_names = prepare_flat_data()
    tr, va = data["train"], data["val"]
    results = []

    models = {
        "LogisticRegression": lambda seed: LogisticRegression(
            max_iter=200, class_weight="balanced", random_state=seed, n_jobs=-1),
        "LinearSVM_SGD": lambda seed: SGDClassifier(
            loss="hinge", class_weight="balanced", random_state=seed, max_iter=50),
        "kNN": lambda seed: KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        "RandomForest": lambda seed: RandomForestClassifier(
            n_estimators=200, max_depth=20, class_weight="balanced", random_state=seed, n_jobs=-1),
        "XGBoost": lambda seed: xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1, objective="multi:softprob",
            num_class=n_classes, random_state=seed, n_jobs=-1, eval_metric="mlogloss"),
        "LightGBM": lambda seed: lgb.LGBMClassifier(
            n_estimators=300, num_leaves=63, learning_rate=0.05, objective="multiclass",
            num_class=n_classes, class_weight="balanced", random_state=seed, verbosity=-1),
        "CatBoost": lambda seed: CatBoostClassifier(
            iterations=300, depth=8, learning_rate=0.1, loss_function="MultiClass",
            auto_class_weights="SqrtBalanced", random_seed=seed, verbose=False),
    }

    for name, factory in models.items():
        for seed in seeds:
            t0 = time.time()
            clf = factory(seed)
            if name == "kNN":
                # kNN prediction cost scales with reference-set size; subsample train
                # for tractability (documented), keep full val/test for evaluation
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(tr["y"]), size=min(30000, len(tr["y"])), replace=False)
                clf.fit(tr["X"][idx], tr["y"][idx])
            elif name == "LinearSVM_SGD":
                clf.fit(tr["X"], tr["y"])
            else:
                clf.fit(tr["X"], tr["y"])
            train_time = time.time() - t0

            t1 = time.time()
            if name == "LinearSVM_SGD":
                # SGDClassifier(hinge) has no predict_proba; use decision_function softmax as a proxy
                scores = clf.decision_function(va["X"])
                if scores.ndim == 1:
                    scores = np.stack([-scores, scores], axis=1)
                proba = np.exp(scores - scores.max(1, keepdims=True))
                proba = proba / proba.sum(1, keepdims=True)
                pred = clf.predict(va["X"])
            else:
                proba = clf.predict_proba(va["X"])
                pred = proba.argmax(1)
            infer_ms = (time.time() - t1) / len(va["y"]) * 1000

            m = eval_metrics(va["y"], pred, proba, n_classes)
            n_params = sum(t.count_params() if hasattr(t, "count_params") else 0 for t in []) or getattr(clf, "n_features_in_", 0)
            m.update(model=name, seed=seed, train_time_sec=train_time, infer_time_ms_per_sample=infer_ms)
            results.append(m)
            print(f"{name} seed={seed}: macroF1={m['macro_F1']:.2f} minRecall={m['min_class_recall']:.2f} "
                  f"train={train_time:.1f}s", flush=True)

    out = pd.DataFrame(results)
    out.to_csv(os.path.join(RESULTS_DIR, "classical_benchmarks_val.csv"), index=False)
    print("Saved results/classical_benchmarks_val.csv")
    return out


if __name__ == "__main__":
    run_all()
