"""Round-2 experiment A: component ablation with ten seeds, reported for standalone SBGA-F and for the
complete stack (each ablated variant stacked with the CatBoost model of the same seed).
Arms: temporal-only (no graph), incoming-only, outgoing-only, no structural descriptors, dense aggregation
(no learned attention), neighbourhood size k in {2, 4, 12, all}, TCN encoder, cross-entropy loss.
Usage: python 31_r2_ablation.py <seed>...   (requires results/probs/<seed>_CatBoost_{val,test}.npy)"""
import os, sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(__file__))
import exp_common as E

OUT = os.path.join(E.RESULTS_DIR, "rev2")
os.makedirs(OUT, exist_ok=True)
ARM_LIST = ["no_graph", "no_bidirectional", "out_only", "no_structural", "dense_aggregation",
            "k2", "k4", "k12", "kall", "TCN_encoder", "CE_loss"]
K_OF = {"k2": 2, "k4": 4, "k12": 12, "kall": 100}


def main():
    seeds = [int(s) for s in sys.argv[1:]]
    base = E.load_base()
    data = E.build_dataset(base, 100)
    X = torch.as_tensor(base.X).to(E.DEVICE)
    va, te = data["splits"]["val"], data["splits"]["test"]
    proto = data["ctx"].proto.cpu().numpy()
    data_k = {a: dict(data, ctx=E.GraphCtx(data["ctx"].A_in, proto, k=k)) for a, k in K_OF.items()}
    for seed in seeds:
        path = os.path.join(OUT, f"ablation_{seed}.csv")
        if os.path.exists(path):
            continue
        cb_va, cb_te = E.load_probs(seed, "CatBoost", "val"), E.load_probs(seed, "CatBoost", "test")
        rows = []
        for arm in ARM_LIST:
            d = data_k.get(arm, data)
            done = os.path.join(E.PROB_DIR, f"{seed}_abl_{arm}_test.npy")
            if os.path.exists(done):        # resume: probabilities of a finished arm are reused
                pv, pt = E.load_probs(seed, f"abl_{arm}", "val"), E.load_probs(seed, f"abl_{arm}", "test")
                pred = type("P", (), dict(n_params=float("nan"), train_time=float("nan")))()
            else:
                pred = E.train_sbgaf(base, d, X, seed, "Full" if arm in K_OF else arm)
                pv, pt = pred.proba(va, d["ctx"], X), pred.proba(te, d["ctx"], X)
                E.save_probs(seed, f"abl_{arm}", "val", pv)
                E.save_probs(seed, f"abl_{arm}", "test", pt)
            meta = E.fit_meta([pv, cb_va], va.y_np)
            ps = meta.predict_proba(np.concatenate([pt, cb_te], axis=1))
            for level, p in (("standalone", pt), ("stack", ps)):
                m = E.full_metrics(te.y_np, p)
                m.update(arm=arm, seed=seed, level=level, n_params=pred.n_params, train_time_sec=pred.train_time)
                rows.append(m)
            print(f"seed {seed} {arm:18s} alone={rows[-2]['macro_F1']:.2f} stack={rows[-1]['macro_F1']:.2f}", flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
