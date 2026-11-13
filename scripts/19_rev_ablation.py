"""Revision experiment 2: component ablation with repeated seeds (all arms except Full,
which is the main-benchmark SBGA-F run of the same seed). Usage: python 19_rev_ablation.py <seed>..."""
import os, sys, time
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(__file__))
import exp_common as E

OUT = os.path.join(E.RESULTS_DIR, "rev")
ARM_LIST = ["no_bidirectional", "no_structural", "dense_aggregation", "TCN_encoder", "CE_loss", "no_graph"]


def main():
    seeds = [int(s) for s in sys.argv[1:]]
    base = E.load_base()
    data = E.build_dataset(base, 100)
    X = torch.as_tensor(base.X).to(E.DEVICE)
    te = data["splits"]["test"]
    for seed in seeds:
        path = os.path.join(OUT, f"ablation_{seed}.csv")
        if os.path.exists(path):
            continue
        rows = []
        for arm in ARM_LIST:
            pred = E.train_sbgaf(base, data, X, seed, arm)
            m = E.full_metrics(te.y_np, pred.proba(te, data["ctx"], X))
            m.update(arm=arm, seed=seed, n_params=pred.n_params, train_time_sec=pred.train_time)
            rows.append(m)
            print(f"seed {seed} {arm:18s} macroF1={m['macro_F1']:.2f} minRec={m['min_class_recall']:.1f}", flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
