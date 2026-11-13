"""Revision experiment 4b: retraining-free transfer with traffic-weighted descriptor
standardization (descriptors z-scored with per-node training-event weights, so their scale does not
depend on how many nodes a graph has). Usage: python 21b_rev_transfer_tw.py <seed>..."""
import os, sys
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
    data = {k: E.build_dataset(base, k, struct_norm="traffic") for k in TARGETS}
    shift = []
    for k in TARGETS:
        te = data[k]["splits"]["test"]
        sf = data[k]["ctx"].struct[te.node].cpu().numpy()
        for j, nm in enumerate(["in_degree", "out_degree", "pagerank", "clustering"]):
            shift.append(dict(system=k, descriptor=nm, traffic_mean=float(sf[:, j].mean()), traffic_std=float(sf[:, j].std())))
    pd.DataFrame(shift).to_csv(os.path.join(OUT, "descriptor_shift_tw.csv"), index=False)
    for seed in seeds:
        path = os.path.join(OUT, f"transfer_tw_{seed}.csv")
        if os.path.exists(path):
            continue
        rows = []
        pred = E.train_sbgaf(base, data[100], X, seed, "Full")
        for k in TARGETS:
            te = data[k]["splits"]["test"]
            m = E.full_metrics(te.y_np, pred.proba(te, data[k]["ctx"], X))
            m.update(seed=seed, model="SBGA-F (traffic-weighted descriptors)", target=k,
                     mode="same_topology" if k == 100 else "transfer", kind="sbgaf")
            rows.append(m)
        print(f"seed {seed} TW " + " ".join(f"{r['target']}:{r['macro_F1']:.2f}" for r in rows), flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
