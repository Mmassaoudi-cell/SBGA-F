"""Round-2 experiment E: equal tuning budget for the strongest baselines.
SBGA-F was tuned with 20 TPE trials on validation macro-F1 (07_tune_finalists.py). The same budget
(20 TPE trials, validation macro-F1, seed 42) is given here to CatBoost, LightGBM, XGBoost, Random Forest,
BiRAGNN and the RNN; the tuned configurations are then run with the ten benchmark seeds and CatBoost-tuned
is stacked with SBGA-F.
Usage: python 35_r2_tune.py tune <model>...      -> results/rev2/tuned_<model>.json
       python 35_r2_tune.py eval <seed>...       -> results/rev2/tuned_<seed>.csv (+ probabilities *_tuned)"""
import json, os, sys, time
import numpy as np
import pandas as pd
import torch
import optuna
sys.path.insert(0, os.path.dirname(__file__))
import exp_common as E

optuna.logging.set_verbosity(optuna.logging.WARNING)
OUT = os.path.join(E.RESULTS_DIR, "rev2")
os.makedirs(OUT, exist_ok=True)
N_TRIALS = 20
TREES = ["CatBoost", "LightGBM", "XGBoost", "RandomForest"]
NEURAL = ["BiRAGNN", "RNN"]


def space(name, t):
    if name == "CatBoost":
        return dict(iterations=t.suggest_categorical("iterations", [300, 500, 800]), depth=t.suggest_int("depth", 4, 9),
                    learning_rate=t.suggest_float("learning_rate", 0.03, 0.3, log=True),
                    l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True))
    if name == "LightGBM":
        return dict(n_estimators=t.suggest_categorical("n_estimators", [300, 500, 800]), num_leaves=t.suggest_int("num_leaves", 15, 255, log=True),
                    learning_rate=t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                    min_child_samples=t.suggest_int("min_child_samples", 5, 100, log=True),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0))
    if name == "XGBoost":
        return dict(n_estimators=t.suggest_categorical("n_estimators", [300, 500, 800]), max_depth=t.suggest_int("max_depth", 4, 12),
                    learning_rate=t.suggest_float("learning_rate", 0.03, 0.3, log=True),
                    subsample=t.suggest_float("subsample", 0.6, 1.0), colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0))
    if name == "RandomForest":
        return dict(n_estimators=t.suggest_categorical("n_estimators", [200, 400]), max_depth=t.suggest_categorical("max_depth", [10, 20, 30, None]),
                    min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 5), max_features=t.suggest_categorical("max_features", ["sqrt", 0.3, 0.5]))
    return dict(hidden=t.suggest_categorical("hidden", [64, 128, 192]), layers=t.suggest_int("layers", 1, 3),
                dropout=t.suggest_float("dropout", 0.1, 0.4), lr=t.suggest_float("lr", 3e-4, 3e-3, log=True))


def setup():
    base = E.load_base()
    data = E.build_dataset(base, 100)
    X = torch.as_tensor(base.X).to(E.DEVICE)
    return base, data, X


def tune(models):
    base, data, X = setup()
    tr, va = data["splits"]["train"], data["splits"]["val"]
    Xtr, Xva = base.X[tr.pos], base.X[va.pos]
    for name in models:
        path = os.path.join(OUT, f"tuned_{name}.json")
        if os.path.exists(path):
            continue
        t0 = time.time()

        def obj(t):
            hp = space(name, t)
            if name in TREES:
                clf = E.fit_classical(name, 42, base, Xtr, tr.y_np, hp)
                p = clf.predict_proba(Xva)
            else:
                p = E.train_ids(base, data, X, name, 42, hp).proba(va, data["ctx"], X)
            return E.macro_f1(va.y_np, p.argmax(1))

        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        st.optimize(obj, n_trials=N_TRIALS)
        json.dump(dict(model=name, n_trials=N_TRIALS, best_val_macro_F1=st.best_value, params=st.best_params,
                       tune_time_sec=time.time() - t0, trials=[dict(v=t.value, p=t.params) for t in st.trials]),
                  open(path, "w"), indent=1)
        print(f"tuned {name}: val macro-F1 {st.best_value:.2f} {st.best_params} ({time.time() - t0:.0f}s)", flush=True)


def evaluate(seeds):
    base, data, X = setup()
    tr, va, te = (data["splits"][s] for s in ("train", "val", "test"))
    Xtr, Xva, Xte = base.X[tr.pos], base.X[va.pos], base.X[te.pos]
    hp = {n: json.load(open(os.path.join(OUT, f"tuned_{n}.json")))["params"] for n in TREES + NEURAL}
    for seed in seeds:
        path = os.path.join(OUT, f"tuned_{seed}.csv")
        if os.path.exists(path):
            continue
        rows, P = [], {}

        def rec(name, pv, pt):
            P[name] = (pv, pt)
            E.save_probs(seed, f"{name}_tuned", "val", pv)
            E.save_probs(seed, f"{name}_tuned", "test", pt)
            m = E.full_metrics(te.y_np, pt)
            m.update(model=name, seed=seed)
            rows.append(m)
            print(f"seed {seed} tuned {name:14s} macroF1={m['macro_F1']:.2f} minRec={m['min_class_recall']:.1f}", flush=True)

        for name in TREES:
            clf = E.fit_classical(name, seed, base, Xtr, tr.y_np, hp[name])
            rec(name, clf.predict_proba(Xva), clf.predict_proba(Xte))
        for name in NEURAL:
            pred = E.train_ids(base, data, X, name, seed, hp[name])
            rec(name, pred.proba(va, data["ctx"], X), pred.proba(te, data["ctx"], X))
        sv, st_ = E.load_probs(seed, "SBGA-F", "val"), E.load_probs(seed, "SBGA-F", "test")
        for a, label in [(None, "SBGA-F + CatBoost-tuned"), ("RNN", "RNN-tuned + CatBoost-tuned"),
                         ("BiRAGNN", "BiRAGNN-tuned + CatBoost-tuned")]:
            av, at = (sv, st_) if a is None else P[a]
            meta = E.fit_meta([av, P["CatBoost"][0]], va.y_np)
            rec(label, meta.predict_proba(np.concatenate([av, P["CatBoost"][0]], axis=1)),
                meta.predict_proba(np.concatenate([at, P["CatBoost"][1]], axis=1)))
        pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    if sys.argv[1] == "tune":
        tune(sys.argv[2:])
    else:
        evaluate([int(s) for s in sys.argv[2:]])
