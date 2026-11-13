"""
Revision experiment 1 (single protocol, 10 seeds): the full 16-model benchmark, SBGA-F-Stack,
and the perturbation suite (feature-type-aware corruptions + topology perturbations).
Usage: python 18_rev_main.py <seed> [<seed> ...]
Outputs: results/rev/main_<seed>.csv, results/rev/perturb_<seed>.csv, results/probs/*.npy
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
import exp_common as E

OUT = os.path.join(E.RESULTS_DIR, "rev")
KEY_PROBS = ["SBGA-F", "CatBoost", "LightGBM", "XGBoost", "RandomForest", "BiRAGNN", "RNN", "GNN", "TGNN"]
ROBUST_TREES = ["CatBoost", "LightGBM", "XGBoost", "RandomForest"]
CLASSICAL = ["LogisticRegression", "LinearSVM_SGD", "kNN", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]
DEEP = ["FNN", "RNN", "CNN", "Transformer", "GNN", "TGNN", "BiRAGNN"]
CATEGORY = {"LogisticRegression": "classical", "LinearSVM_SGD": "classical", "kNN": "classical",
            "RandomForest": "tree", "XGBoost": "tree", "LightGBM": "tree", "CatBoost": "tree",
            "FNN": "deep", "RNN": "deep", "CNN": "deep", "Transformer": "deep", "FT-Transformer": "deep",
            "GNN": "graph", "TGNN": "graph", "BiRAGNN": "graph", "SBGA-F": "proposed",
            "SBGA-F-Stack": "proposed_stack"}

PERTURBATIONS = (
    [("feature", "num_noise", s) for s in (0.1, 0.25, 0.5, 1.0)]
    + [("feature", "cat_corrupt", p) for p in (0.05, 0.10, 0.25)]
    + [("feature", "missing", p) for p in (0.10, 0.25, 0.50)]
    + [("feature", "flow_dropout", 1.0)]
    + [("feature", "jitter", s) for s in (0.25, 0.5)]
    + [("feature", "combined", 1.0)]
    + [("topology", "edge_drop", p) for p in (0.1, 0.3, 0.5)]
    + [("topology", "weight_noise", 0.5)]
    + [("topology", "rewire", p) for p in (0.1, 0.3)]
)


def run_seed(seed, base, data, X_gpu):
    rows, models, probs = [], {}, {}
    tr, va, te = (data["splits"][s] for s in ("train", "val", "test"))
    ctx = data["ctx"]
    Xtr, Xva, Xte = (E.flat_split(base, s) for s in (tr, va, te))

    def record(name, p_val, p_te, train_time, infer_ms, n_params=None):
        m = E.full_metrics(te.y_np, p_te)
        m.update(model=name, seed=seed, category=CATEGORY.get(name, "other"),
                 train_time_sec=train_time, infer_ms_per_sample=infer_ms, n_params=n_params)
        rows.append(m)
        probs[name] = (p_val, p_te)
        print(f"  seed {seed} {name:20s} macroF1={m['macro_F1']:.2f} minRec={m['min_class_recall']:.1f}", flush=True)

    # ---- classical / tree models
    for name in CLASSICAL:
        clf = E.fit_classical(name, seed, base, Xtr, tr.y_np)
        t0 = time.time()
        p_te = E.classical_proba(clf, name, Xte)
        infer = (time.time() - t0) / len(Xte) * 1000
        p_va = E.classical_proba(clf, name, Xva)
        record(name, p_va, p_te, clf.train_time, infer)
        if name in ROBUST_TREES:
            models[name] = clf

    # ---- neural / graph baselines
    for name in DEEP:
        pred = E.train_ids(base, data, X_gpu, name, seed)
        t0 = time.time()
        p_te = pred.proba(te, ctx, X_gpu)
        torch.cuda.synchronize()
        infer = (time.time() - t0) / te.n * 1000
        record(name, pred.proba(va, ctx, X_gpu), p_te, pred.train_time, infer, pred.n_params)
        if name in ("BiRAGNN", "RNN"):
            models[name] = pred
    pred = E.train_ft(base, data, X_gpu, seed)
    t0 = time.time()
    p_te = pred.proba(te, ctx, X_gpu)
    torch.cuda.synchronize()
    record("FT-Transformer", pred.proba(va, ctx, X_gpu), p_te, pred.train_time,
           (time.time() - t0) / te.n * 1000, pred.n_params)

    # ---- proposed
    pred = E.train_sbgaf(base, data, X_gpu, seed, "Full")
    t0 = time.time()
    p_te = pred.proba(te, ctx, X_gpu)
    torch.cuda.synchronize()
    record("SBGA-F", pred.proba(va, ctx, X_gpu), p_te, pred.train_time,
           (time.time() - t0) / te.n * 1000, pred.n_params)
    models["SBGA-F"] = pred

    # ---- stack (logistic-regression meta-learner fitted on validation probabilities)
    meta = E.fit_meta([probs["SBGA-F"][0], probs["CatBoost"][0]], va.y_np)
    p_stack_va = meta.predict_proba(np.concatenate([probs["SBGA-F"][0], probs["CatBoost"][0]], axis=1))
    p_stack_te = meta.predict_proba(np.concatenate([probs["SBGA-F"][1], probs["CatBoost"][1]], axis=1))
    t_sb = [r for r in rows if r["model"] == "SBGA-F"][0]["train_time_sec"]
    t_cb = [r for r in rows if r["model"] == "CatBoost"][0]["train_time_sec"]
    record("SBGA-F-Stack", p_stack_va, p_stack_te, t_sb + t_cb, float("nan"))

    for n in KEY_PROBS + ["SBGA-F-Stack"]:
        E.save_probs(seed, n, "val", probs[n][0])
        E.save_probs(seed, n, "test", probs[n][1])
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"main_{seed}.csv"), index=False)

    # ---- perturbation suite (models are frozen; perturbations applied to test events only)
    prow = []

    def prec(model, fam, kind, level, p):
        pr = p.argmax(1)
        prow.append(dict(seed=seed, model=model, family=fam, perturbation=kind, level=level,
                         macro_F1=E.macro_f1(te.y_np, pr), min_recall=E.min_recall(te.y_np, pr)))

    proto_cpu = ctx.proto.cpu().numpy()
    sb, bi, rnn = models["SBGA-F"], models["BiRAGNN"], models["RNN"]
    for fam, kind, level in [("feature", "clean", 0.0)] + PERTURBATIONS:
        if fam == "feature":
            Xp = E.perturb_features(base, kind, level, seed)
            Xp_gpu = torch.as_tensor(Xp).to(E.DEVICE)
            Xte_p = Xp[te.pos]
            ps = {"SBGA-F": sb.proba(te, ctx, Xp_gpu), "BiRAGNN": bi.proba(te, ctx, Xp_gpu),
                  "RNN": rnn.proba(te, ctx, Xp_gpu)}
            for tname in ROBUST_TREES:
                ps[tname] = E.classical_proba(models[tname], tname, Xte_p)
            ps["SBGA-F-Stack"] = meta.predict_proba(np.concatenate([ps["SBGA-F"], ps["CatBoost"]], axis=1))
        else:
            A_p = E.perturb_adjacency(ctx.A_in, kind, level, seed)
            ctx_p = E.GraphCtx(A_p, proto_cpu)
            ps = {"SBGA-F": sb.proba(te, ctx_p, X_gpu), "BiRAGNN": bi.proba(te, ctx_p, X_gpu)}
            ps["SBGA-F-Stack"] = meta.predict_proba(np.concatenate([ps["SBGA-F"], probs["CatBoost"][1]], axis=1))
        for mname, p in ps.items():
            prec(mname, fam, kind, level, p)
    pd.DataFrame(prow).to_csv(os.path.join(OUT, f"perturb_{seed}.csv"), index=False)


def main():
    seeds = [int(s) for s in sys.argv[1:]]
    os.makedirs(OUT, exist_ok=True)
    base = E.load_base()
    data = E.build_dataset(base, 100, "transition", "original")
    X_gpu = torch.as_tensor(base.X).to(E.DEVICE)
    for seed in seeds:
        if os.path.exists(os.path.join(OUT, f"perturb_{seed}.csv")):
            print(f"seed {seed} already done")
            continue
        t0 = time.time()
        run_seed(seed, base, data, X_gpu)
        print(f"== seed {seed} total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
