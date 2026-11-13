"""Round-2 experiment D: robustness controls, ten seeds.
 * degraded graph context: prototypes recomputed from equally corrupted historical (training-partition) features;
   stale graph: adjacency and prototypes from the first 10% / 1% of the training stream;
 * noise- and missingness-augmented CatBoost / LightGBM, and the stack built on the augmented CatBoost;
 * repeated perturbation draws (five per seed) with realised removal counts;
 * gradient-based adversarial perturbation (FGSM, L-infinity on numeric features, standardized units).
Usage: python 34_r2_robust.py <seed>..."""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import exp_common as E

OUT = os.path.join(E.RESULTS_DIR, "rev2")
os.makedirs(OUT, exist_ok=True)
FEATURE = [("num_noise", 0.25), ("num_noise", 0.5), ("missing", 0.10), ("missing", 0.25), ("cat_corrupt", 0.10),
           ("flow_dropout", 1.0), ("jitter", 0.5), ("combined", 1.0)]
CTX = [("num_noise", 0.25), ("num_noise", 0.5), ("missing", 0.10), ("missing", 0.25), ("combined", 1.0)]
DRAWS = [("num_noise", 0.25), ("missing", 0.10)]
EPS = [0.05, 0.1, 0.2]


def augment(base, Xtr, ytr, seed):
    rng = np.random.default_rng(5000 + seed)
    n, d = Xtr.shape
    a = Xtr.copy()
    a[:, base.num_idx] += (rng.normal(0, 1, size=(n, len(base.num_idx))) * rng.uniform(0, 0.5, size=(n, 1))).astype(np.float32)
    b = np.where(rng.random((n, d)) < rng.uniform(0, 0.3, size=(n, 1)), base.fill_std[None, :], Xtr).astype(np.float32)
    return np.concatenate([Xtr, a, b]), np.concatenate([ytr, ytr, ytr])


def fgsm(pred, te, ctx, X_gpu, eps, num_idx, chunk=8192):
    Xa = X_gpu.clone().requires_grad_(True)
    pred.model.train()          # cuDNN LSTM needs train mode for backward; dropout is disabled below
    for m in pred.model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
        if isinstance(m, torch.nn.LSTM):
            m.dropout = 0.0
    g = torch.zeros_like(X_gpu)
    for i in range(0, te.n, chunk):
        idx = torch.arange(i, min(i + chunk, te.n), device=E.DEVICE)
        loss = F.cross_entropy(pred.logits(te, ctx, Xa, idx), te.y[idx], reduction="sum")
        g += torch.autograd.grad(loss, Xa)[0]
    pred.model.eval()
    step = torch.zeros_like(X_gpu)
    step[:, num_idx] = eps * g[:, num_idx].sign()
    return (X_gpu + step).detach()


def main():
    seeds = [int(s) for s in sys.argv[1:]]
    base = E.load_base()
    X = torch.as_tensor(base.X).to(E.DEVICE)
    data = E.build_dataset(base, 100)
    tr, va, te = (data["splits"][s] for s in ("train", "val", "test"))
    ctx = data["ctx"]
    Xtr, Xva = base.X[tr.pos], base.X[va.pos]
    node = base.df["node_100"].to_numpy().astype(np.int64)
    stale = {}
    for f in (0.1, 0.01):
        pos = tr.pos[: int(len(tr.pos) * f)]
        A = E.adjacency_from_pairs(node[pos][:-1], node[pos][1:], 100)
        stale[f] = E.GraphCtx(A, E.compute_prototypes(base.X[pos], node[pos], 100)[0])
    for seed in seeds:
        path = os.path.join(OUT, f"robust_{seed}.csv")
        if os.path.exists(path):
            continue
        rows = []
        sb = E.train_sbgaf(base, data, X, seed, "Full")
        bi = E.train_ids(base, data, X, "BiRAGNN", seed)
        rnn = E.train_ids(base, data, X, "RNN", seed)
        trees = {n: E.fit_classical(n, seed, base, Xtr, tr.y_np) for n in ("CatBoost", "LightGBM")}
        Xa, ya = augment(base, Xtr, tr.y_np, seed)
        for n in ("CatBoost", "LightGBM"):
            trees[n + "-aug"] = E.fit_classical(n, seed, base, Xa, ya)
        sb_va = sb.proba(va, ctx, X)
        meta = E.fit_meta([sb_va, trees["CatBoost"].predict_proba(Xva)], va.y_np)
        meta_aug = E.fit_meta([sb_va, trees["CatBoost-aug"].predict_proba(Xva)], va.y_np)

        def evaluate(tag, kind, level, Xp_gpu, c, draw=0, extra=None, models="all"):
            Xte_p = Xp_gpu[torch.as_tensor(te.pos, device=E.DEVICE)].cpu().numpy()
            ps = {"SBGA-F": sb.proba(te, c, Xp_gpu)}
            if models == "all":
                ps["BiRAGNN"] = bi.proba(te, c, Xp_gpu)
                ps["RNN"] = rnn.proba(te, c, Xp_gpu)
                for n, clf in trees.items():
                    ps[n] = clf.predict_proba(Xte_p)
            else:
                ps["CatBoost"] = trees["CatBoost"].predict_proba(Xte_p)
            ps["SBGA-F-Stack"] = meta.predict_proba(np.concatenate([ps["SBGA-F"], ps["CatBoost"]], axis=1))
            if "CatBoost-aug" in ps:
                ps["SBGA-F-Stack-aug"] = meta_aug.predict_proba(np.concatenate([ps["SBGA-F"], ps["CatBoost-aug"]], axis=1))
            for mname, p in ps.items():
                pr = p.argmax(1)
                r = dict(seed=seed, condition=tag, perturbation=kind, level=level, draw=draw, model=mname,
                         macro_F1=E.macro_f1(te.y_np, pr), min_recall=E.min_recall(te.y_np, pr))
                r.update(extra or {})
                rows.append(r)

        evaluate("clean", "clean", 0.0, X, ctx)
        for f, c in stale.items():
            evaluate("stale_graph", "stale", f, X, c)
        for kind, level in FEATURE:
            Xp = E.perturb_features(base, kind, level, seed)
            evaluate("feature", kind, level, torch.as_tensor(Xp).to(E.DEVICE), ctx)
            if (kind, level) in CTX:
                Xh = E.perturb_features(base, kind, level, seed + 777, split="train")
                proto_p, _ = E.compute_prototypes(Xh[tr.pos], tr.node_np, 100)
                c_p = E.GraphCtx(ctx.A_in, proto_p)
                evaluate("feature+context", kind, level, torch.as_tensor(Xp).to(E.DEVICE), c_p)
                for f in (0.1,):
                    pos = tr.pos[: int(len(tr.pos) * f)]
                    c_s = E.GraphCtx(stale[f].A_in, E.compute_prototypes(Xh[pos], node[pos], 100)[0])
                    evaluate("feature+context+stale", kind, level, torch.as_tensor(Xp).to(E.DEVICE), c_s)
        for kind, level in DRAWS:
            for draw in range(1, 5):
                Xp = E.perturb_features(base, kind, level, seed + 100 * draw)
                changed = (Xp[te.pos] != base.X[te.pos])
                evaluate("draws", kind, level, torch.as_tensor(Xp).to(E.DEVICE), ctx, draw=draw, models="few",
                         extra=dict(cells_changed=int(changed.sum()), frac_changed=float(changed.mean()),
                                    rows_changed=int(changed.any(1).sum())))
        for eps in EPS:
            for sname, sur in (("SBGA-F", sb), ("BiRAGNN", bi), ("RNN", rnn)):
                Xadv = fgsm(sur, te, ctx, X, eps, base.num_idx)
                if sname == "SBGA-F":
                    evaluate("fgsm_from_SBGA-F", "fgsm", eps, Xadv, ctx)
                else:
                    pr = sur.proba(te, ctx, Xadv).argmax(1)
                    rows.append(dict(seed=seed, condition="fgsm_whitebox", perturbation="fgsm", level=eps, draw=0, model=sname,
                                     macro_F1=E.macro_f1(te.y_np, pr), min_recall=E.min_recall(te.y_np, pr)))
        pd.DataFrame(rows).to_csv(path, index=False)
        df = pd.DataFrame(rows)
        print(f"== seed {seed}\n" + df[df.draw == 0].pivot_table(index=["condition", "perturbation", "level"], columns="model",
              values="macro_F1").round(1).to_string(), flush=True)


if __name__ == "__main__":
    main()
