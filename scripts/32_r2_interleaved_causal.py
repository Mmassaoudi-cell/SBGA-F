"""Round-2 experiment B: two leakage controls with ten seeds.
  mode "interleaved": class-interleaved event stream (windows mix classes). Trains SBGA-F, the temporal-only
        variant, the RNN and BiRAGNN and stores validation/test probabilities in stream order.
  mode "causal": flow aggregates recomputed causally (only packets up to the current one) with a flow key
        that does not contain the class label. Trains CatBoost, LightGBM, RNN, BiRAGNN, SBGA-F and the stack.
Usage: python 32_r2_interleaved_causal.py <mode> <seed>..."""
import os, sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(__file__))
import exp_common as E

OUT = os.path.join(E.RESULTS_DIR, "rev2")
os.makedirs(OUT, exist_ok=True)


def run_interleaved(seeds):
    base = E.load_base()
    X = torch.as_tensor(base.X).to(E.DEVICE)
    d = E.build_dataset(base, 100, graph="transition", stream="interleaved")
    va, te = d["splits"]["val"], d["splits"]["test"]
    for seed in seeds:
        if os.path.exists(os.path.join(E.PROB_DIR, f"{seed}_SBGA-F_no_graph_interleaved_test.npy")):
            continue
        for mname in ["SBGA-F", "BiRAGNN", "RNN", "SBGA-F_no_graph"]:
            if mname == "SBGA-F":
                pred = E.train_sbgaf(base, d, X, seed, "Full")
            elif mname == "SBGA-F_no_graph":
                pred = E.train_sbgaf(base, d, X, seed, "no_graph")
            else:
                pred = E.train_ids(base, d, X, mname, seed)
            p = pred.proba(te, d["ctx"], X)
            E.save_probs(seed, f"{mname}_interleaved", "val", pred.proba(va, d["ctx"], X))
            E.save_probs(seed, f"{mname}_interleaved", "test", p)
            print(f"seed {seed} interleaved {mname:16s} macroF1={E.macro_f1(te.y_np, p.argmax(1)):.2f}", flush=True)


def run_causal(seeds):
    base = E.load_base("causal")
    X = torch.as_tensor(base.X).to(E.DEVICE)
    d = E.build_dataset(base, 100)
    tr, va, te = (d["splits"][s] for s in ("train", "val", "test"))
    Xtr, Xva, Xte = base.X[tr.pos], base.X[va.pos], base.X[te.pos]
    for seed in seeds:
        path = os.path.join(OUT, f"causal_{seed}.csv")
        if os.path.exists(path):
            continue
        rows, P = [], {}

        def rec(name, pv, pt):
            P[name] = (pv, pt)
            E.save_probs(seed, f"{name}_causal", "val", pv)
            E.save_probs(seed, f"{name}_causal", "test", pt)
            m = E.full_metrics(te.y_np, pt)
            m.update(model=name, seed=seed)
            rows.append(m)
            print(f"seed {seed} causal {name:14s} macroF1={m['macro_F1']:.2f} minRec={m['min_class_recall']:.1f}", flush=True)

        for name in ["CatBoost", "LightGBM"]:
            clf = E.fit_classical(name, seed, base, Xtr, tr.y_np)
            rec(name, clf.predict_proba(Xva), clf.predict_proba(Xte))
        for name in ["RNN", "BiRAGNN"]:
            pred = E.train_ids(base, d, X, name, seed)
            rec(name, pred.proba(va, d["ctx"], X), pred.proba(te, d["ctx"], X))
        for arm, name in [("Full", "SBGA-F"), ("no_graph", "SBGA-F_no_graph")]:
            pred = E.train_sbgaf(base, d, X, seed, arm)
            rec(name, pred.proba(va, d["ctx"], X), pred.proba(te, d["ctx"], X))
        for a, label in [("SBGA-F", "SBGA-F-Stack"), ("RNN", "RNN+CatBoost"), ("SBGA-F_no_graph", "NoGraph+CatBoost")]:
            meta = E.fit_meta([P[a][0], P["CatBoost"][0]], va.y_np)
            rec(label, meta.predict_proba(np.concatenate([P[a][0], P["CatBoost"][0]], axis=1)),
                meta.predict_proba(np.concatenate([P[a][1], P["CatBoost"][1]], axis=1)))
        pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    mode, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
    (run_interleaved if mode == "interleaved" else run_causal)(seeds)
