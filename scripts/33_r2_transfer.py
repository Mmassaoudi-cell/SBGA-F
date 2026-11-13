"""Round-2 experiment C: cross-topology reuse, ten seeds.
 (a) target-data budget: graph statistics and prototypes of the 50-/8-node target recomputed from the first
     fraction f of the unlabeled target training stream, f in {0.01, 0.1, 1};
 (b) temporal-only variant, BiRAGNN, and the complete stack under the same target conditions;
 (c) node-indexed embedding baseline (parameters tied to the 100 training nodes);
 (d) unseen devices: 25 of the 100 nodes are withheld from training and validation; the detector is then
     applied to test events of the withheld nodes after the graph context is rebuilt from unlabeled events.
Usage: python 33_r2_transfer.py <seed>..."""
import os, sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(__file__))
import exp_common as E

OUT = os.path.join(E.RESULTS_DIR, "rev2")
os.makedirs(OUT, exist_ok=True)
TARGETS = [100, 50, 8]
FRACS = [0.01, 0.1, 1.0]


def ctx_from_prefix(base, system, frac):
    node = base.df[f"node_{system}"].to_numpy().astype(np.int64)
    pos = base.split_pos["train"]
    pos = pos[: max(2, int(round(len(pos) * frac)))]
    seq = node[pos]
    A = E.adjacency_from_pairs(seq[:-1], seq[1:], system)
    proto, cnt = E.compute_prototypes(base.X[pos], seq, system)
    return E.GraphCtx(A, proto), int((cnt > 0).sum()), len(pos)


def make_data(base, system, pos_by_split, ctx):
    node_all = base.df[f"node_{system}"].to_numpy().astype(np.int64)
    ds = {}
    for s, pos in pos_by_split.items():
        sp = E.Split()
        sp.pos, sp.node_np = pos, node_all[pos]
        sp.node = E.to_t(sp.node_np, torch.long).to(E.DEVICE)
        sp.win = E.to_t(E.build_windows_fast(pos, sp.node_np), torch.long).to(E.DEVICE)
        sp.y_np = base.y[pos]
        sp.y = E.to_t(sp.y_np, torch.long).to(E.DEVICE)
        sp.n = len(pos)
        ds[s] = sp
    return dict(splits=ds, ctx=ctx, system=system)


def slice_metrics(y, p, mask, min_n=20):
    y, pr = y[mask], p[mask].argmax(1)
    cls = [c for c in np.unique(y) if (y == c).sum() >= min_n]
    from sklearn.metrics import f1_score
    return dict(n=int(mask.sum()), n_classes=len(cls), accuracy=float((y == pr).mean()) * 100,
                macro_F1=f1_score(y, pr, labels=cls, average="macro", zero_division=0) * 100)


def main():
    seeds = [int(s) for s in sys.argv[1:]]
    base = E.load_base()
    X = torch.as_tensor(base.X).to(E.DEVICE)
    data = {k: E.build_dataset(base, k) for k in TARGETS}
    ctxf = {(k, f): ctx_from_prefix(base, k, f) for k in (50, 8) for f in FRACS}
    pd.DataFrame([dict(target=k, frac=f, nodes_with_events=c[1], unlabeled_events=c[2], n_edges=c[0].n_edges)
                  for (k, f), c in ctxf.items()]).to_csv(os.path.join(OUT, "transfer_budget_stats.csv"), index=False)
    node100 = base.df["node_100"].to_numpy().astype(np.int64)
    for seed in seeds:
        path = os.path.join(OUT, f"transfer_{seed}.csv")
        if os.path.exists(path):
            continue
        rows = []
        cb_va, cb_te = E.load_probs(seed, "CatBoost", "val"), E.load_probs(seed, "CatBoost", "test")
        va100 = data[100]["splits"]["val"]

        def ev(pred, label, stack=False):
            meta = E.fit_meta([pred.proba(va100, data[100]["ctx"], X), cb_va], va100.y_np) if stack else None
            for k in TARGETS:
                te = data[k]["splits"]["test"]
                for f in (FRACS if k != 100 else [1.0]):
                    ctx = data[k]["ctx"] if (k == 100 or f == 1.0) else ctxf[(k, f)][0]
                    p = pred.proba(te, ctx, X)
                    m = E.full_metrics(te.y_np, p)
                    m.update(seed=seed, model=label, target=k, frac=f)
                    rows.append(m)
                    if stack:
                        m2 = E.full_metrics(te.y_np, meta.predict_proba(np.concatenate([p, cb_te], axis=1)))
                        m2.update(seed=seed, model=label + "-Stack", target=k, frac=f)
                        rows.append(m2)
            print(f"seed {seed} {label:18s} " + " ".join(f"{r['target']}@{r['frac']}:{r['macro_F1']:.2f}"
                  for r in rows if r["model"] == label and r["seed"] == seed), flush=True)

        ev(E.train_sbgaf(base, data[100], X, seed, "Full"), "SBGA-F", stack=True)
        ev(E.train_sbgaf(base, data[100], X, seed, "no_graph"), "SBGA-F temporal-only")
        ev(E.train_ids(base, data[100], X, "BiRAGNN", seed), "BiRAGNN")
        ev(E.train_nodeemb(base, data[100], X, seed), "Node-embedding LSTM")
        pd.DataFrame(rows).to_csv(path, index=False)

        # ---- (d) unseen devices
        rng = np.random.default_rng(seed)
        ytr, ntr = base.y[base.split_pos["train"]], node100[base.split_pos["train"]]
        for _ in range(500):
            H = rng.choice(100, size=25, replace=False)
            keep = ~np.isin(ntr, H)
            share = np.array([keep[ytr == c].mean() for c in range(base.n_classes)])
            if share.min() >= 0.5:
                break
        pos = {s: base.split_pos[s][~np.isin(node100[base.split_pos[s]], H)] for s in ("train", "val")}
        pos["test"] = base.split_pos["test"]
        seq = node100[pos["train"]]
        A = E.adjacency_from_pairs(seq[:-1], seq[1:], 100)
        proto, _ = E.compute_prototypes(base.X[pos["train"]], seq, 100)
        d = make_data(base, 100, pos, E.GraphCtx(A, proto))
        te = d["splits"]["test"]
        unseen = np.isin(te.node_np, H)
        full_ctx = data[100]["ctx"]            # rebuilt from the unlabeled events of all nodes
        urows = []

        def uev(label, p, ctxname):
            for nm, mask in (("unseen nodes", unseen), ("seen nodes", ~unseen)):
                m = slice_metrics(te.y_np, p, mask)
                m.update(seed=seed, model=label, context=ctxname, slice=nm, min_class_share_kept=float(share.min()))
                urows.append(m)

        for arm, label in [("Full", "SBGA-F"), ("no_graph", "SBGA-F temporal-only")]:
            pred = E.train_sbgaf(base, d, X, seed, arm)
            uev(label, pred.proba(te, full_ctx, X), "rebuilt")
            if arm == "Full":
                uev(label, pred.proba(te, d["ctx"], X), "training graph (no update)")
        for name in ["BiRAGNN", "RNN"]:
            pred = E.train_ids(base, d, X, name, seed)
            uev(name, pred.proba(te, full_ctx, X), "rebuilt")
        clf = E.fit_classical("CatBoost", seed, base, base.X[pos["train"]], base.y[pos["train"]])
        uev("CatBoost", clf.predict_proba(base.X[te.pos]), "n/a")
        pd.DataFrame(urows).to_csv(os.path.join(OUT, f"unseen_{seed}.csv"), index=False)
        print(f"seed {seed} unseen-device " + " ".join(f"{r['model']}[{r['slice'][:2]}]:{r['macro_F1']:.1f}" for r in urows), flush=True)


if __name__ == "__main__":
    main()
