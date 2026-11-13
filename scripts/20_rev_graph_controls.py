"""Revision experiment 3: graph-construction / capture-order controls.
 (a) alternative graphs under the capture-order stream: block-shuffled and fully shuffled event
     order, observed communication-link graph, degree-preserving rewiring;
 (b) interleaved-stream protocol: windows AND graph built from a class-mixed event stream.
Models: SBGA-F and BiRAGNN (graph-based) plus RNN (no graph) for the interleaved protocol.
Usage: python 20_rev_graph_controls.py <seed>..."""
import os, sys, time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(__file__))
import exp_common as E

OUT = os.path.join(E.RESULTS_DIR, "rev")


def main():
    seeds = [int(s) for s in sys.argv[1:]]
    base = E.load_base()
    X = torch.as_tensor(base.X).to(E.DEVICE)
    variants = [("original", "block_shuffle"), ("original", "shuffle"), ("original", "link"),
                ("original", "rewired"), ("interleaved", "transition")]
    datasets = {v: E.build_dataset(base, 100, graph=v[1], stream=v[0]) for v in variants}
    graph_stats = []
    for v, d in datasets.items():
        c = d["ctx"]
        graph_stats.append(dict(stream=v[0], graph=v[1], n_edges=c.n_edges, frac_nodes_lt_k=c.frac_nodes_lt_k_in,
                                n_isolated=c.n_isolated))
    pd.DataFrame(graph_stats).to_csv(os.path.join(OUT, "graph_variant_stats.csv"), index=False)
    for seed in seeds:
        path = os.path.join(OUT, f"graphctl_{seed}.csv")
        if os.path.exists(path):
            continue
        rows = []
        for v in variants:
            d = datasets[v]
            te = d["splits"]["test"]
            models = ["SBGA-F", "BiRAGNN"] + (["RNN", "SBGA-F_no_graph"] if v[0] == "interleaved" else [])
            for mname in models:
                if mname == "SBGA-F":
                    pred = E.train_sbgaf(base, d, X, seed, "Full")
                elif mname == "SBGA-F_no_graph":
                    pred = E.train_sbgaf(base, d, X, seed, "no_graph")
                else:
                    pred = E.train_ids(base, d, X, mname, seed)
                p = pred.proba(te, d["ctx"], X)
                m = E.full_metrics(te.y_np, p)
                m.update(stream=v[0], graph=v[1], model=mname, seed=seed)
                rows.append(m)
                if v[0] == "interleaved":
                    E.save_probs(seed, f"{mname}_interleaved", "test", p)
                    E.save_probs(seed, f"{mname}_interleaved", "val", pred.proba(d["splits"]["val"], d["ctx"], X))
                print(f"seed {seed} {v} {mname:16s} macroF1={m['macro_F1']:.2f} minRec={m['min_class_recall']:.1f}", flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
