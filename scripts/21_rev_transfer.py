"""Revision experiment 4: retraining-free topology transfer, 10 seeds. Trains on the 100-node graph
and evaluates on 100/50/8-node targets whose adjacency, descriptors and prototypes are recomputed
from the target's own training partition. Arms: Full, no_structural, no_bidirectional; baselines
BiRAGNN/TGNN/GNN/RNN; reference: SBGA-F retrained on the target topology.
Usage: python 21_rev_transfer.py <seed>..."""
import os, sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(__file__))
import exp_common as E

OUT = os.path.join(E.RESULTS_DIR, "rev")
TARGETS = [100, 50, 8]


def main():
    seeds = [int(s) for s in sys.argv[1:]]
    base = E.load_base()
    X = torch.as_tensor(base.X).to(E.DEVICE)
    data = {k: E.build_dataset(base, k) for k in TARGETS}
    # descriptor-shift statistics (traffic-weighted), seed independent
    shift = []
    for k in TARGETS:
        te = data[k]["splits"]["test"]
        sf = data[k]["ctx"].struct[te.node].cpu().numpy()
        for j, nm in enumerate(["in_degree", "out_degree", "pagerank", "clustering"]):
            shift.append(dict(system=k, descriptor=nm, traffic_mean=float(sf[:, j].mean()), traffic_std=float(sf[:, j].std()),
                              traffic_max=float(sf[:, j].max())))
    pd.DataFrame(shift).to_csv(os.path.join(OUT, "descriptor_shift.csv"), index=False)
    for seed in seeds:
        path = os.path.join(OUT, f"transfer_{seed}.csv")
        if os.path.exists(path):
            continue
        rows = []

        def ev(pred, label, kind):
            for k in TARGETS:
                te = data[k]["splits"]["test"]
                m = E.full_metrics(te.y_np, pred.proba(te, data[k]["ctx"], X))
                m.update(seed=seed, model=label, target=k, mode="same_topology" if k == 100 else "transfer", kind=kind)
                rows.append(m)
            print(f"seed {seed} {label:20s} " + " ".join(f"{r['target']}:{r['macro_F1']:.2f}" for r in rows[-3:]), flush=True)

        for arm, label in [("Full", "SBGA-F"), ("no_structural", "SBGA-F -struct"), ("no_bidirectional", "SBGA-F -bidir")]:
            ev(E.train_sbgaf(base, data[100], X, seed, arm), label, "sbgaf")
        for name in ["BiRAGNN", "RNN"]:
            ev(E.train_ids(base, data[100], X, name, seed), name, "baseline")
        for k in (50, 8):
            pred = E.train_sbgaf(base, data[k], X, seed, "Full")
            te = data[k]["splits"]["test"]
            m = E.full_metrics(te.y_np, pred.proba(te, data[k]["ctx"], X))
            m.update(seed=seed, model="SBGA-F retrained on target", target=k, mode="retrained", kind="reference")
            rows.append(m)
            print(f"seed {seed} retrained on {k}: {m['macro_F1']:.2f}", flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
