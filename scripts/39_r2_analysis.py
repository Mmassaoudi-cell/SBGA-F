"""Round-2 analysis: aggregates results/rev2/*.csv into paired statistics and prints the summaries used
for the manuscript tables. Usage: python 39_r2_analysis.py [section ...]  (default: all)"""
import glob, os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
R1 = os.path.join(ROOT, "results", "rev")
R2 = os.path.join(ROOT, "results", "rev2")
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)


def read(dirn, prefix):
    fs = [f for f in sorted(glob.glob(os.path.join(dirn, f"{prefix}_*.csv"))) if os.path.basename(f)[len(prefix) + 1:-4].isdigit()]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def paired(a, b):
    """a, b: Series indexed by seed. Returns n, mean diff (a-b), 95% t-CI, exact Wilcoxon p."""
    j = a.index.intersection(b.index)
    d = (a[j] - b[j]).to_numpy()
    n = len(d)
    if n < 2:
        return dict(n=n, diff=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    try:
        p = stats.wilcoxon(d, mode="exact").pvalue if np.any(d != 0) else 1.0
    except Exception:
        p = stats.wilcoxon(d).pvalue
    return dict(n=n, diff=d.mean(), lo=d.mean() - t * se, hi=d.mean() + t * se, p=p)


def ms(s):
    return f"{s.mean():.2f}+-{s.std(ddof=1):.2f}" if len(s) > 1 else f"{s.mean():.2f}"


def fmt(r):
    return f"{r['diff']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}] p={r['p']:.3f} (n={r['n']})"


main = read(R1, "main")
M = {m: g.set_index("seed")["macro_F1"] for m, g in main.groupby("model")}
MR = {m: g.set_index("seed")["min_class_recall"] for m, g in main.groupby("model")}


def sec_ablation():
    d = read(R2, "ablation")
    if d.empty:
        return
    print("\n===== ABLATION (paired vs Full of the same seed) =====")
    for level, ref, refmr in (("standalone", M["SBGA-F"], MR["SBGA-F"]), ("stack", M["SBGA-F-Stack"], MR["SBGA-F-Stack"])):
        seeds = sorted(d.seed.unique())
        print(f"-- {level}: Full = {ms(ref[seeds])}  minrec {ms(refmr[seeds])}  (seeds {len(seeds)})")
        for arm, g in d[d.level == level].groupby("arm", sort=False):
            s = g.set_index("seed")["macro_F1"]
            print(f"   {arm:20s} {ms(s):14s} minrec {ms(g.set_index('seed')['min_class_recall']):12s} vs Full {fmt(paired(s, ref))}"
                  + (f" | vs CatBoost {fmt(paired(s, M['CatBoost']))}" if level == "stack" else ""))
    st = d[d.level == "stack"]
    if "no_graph" in set(st.arm):
        a = st[st.arm == "no_graph"].set_index("seed")["macro_F1"]
        print("   temporal-only stack vs RNN stack: see stack controls; vs CatBoost", fmt(paired(a, M["CatBoost"])))


def sec_tuned():
    d = read(R2, "tuned")
    if d.empty:
        return
    print("\n===== EQUAL-BUDGET TUNING =====")
    import json
    for f in sorted(glob.glob(os.path.join(R2, "tuned_*.json"))):
        j = json.load(open(f))
        print(f"   {j['model']:13s} best val {j['best_val_macro_F1']:.2f} time {j['tune_time_sec']:.0f}s {j['params']}")
    T = {m: g.set_index("seed")["macro_F1"] for m, g in d.groupby("model")}
    TR = {m: g.set_index("seed")["min_class_recall"] for m, g in d.groupby("model")}
    for m in T:
        line = f"   {m:32s} tuned {ms(T[m]):12s} minrec {ms(TR[m])}"
        if m in M:
            line += f" | fixed {ms(M[m])} | tuned-fixed {fmt(paired(T[m], M[m]))} | SBGA-F - tuned {fmt(paired(M['SBGA-F'], T[m]))}"
        else:
            line += f" | vs CatBoost-tuned {fmt(paired(T[m], T['CatBoost']))}"
        print(line)
    print("   SBGA-F-Stack(fixed CatBoost) - CatBoost-tuned", fmt(paired(M["SBGA-F-Stack"], T["CatBoost"])))
    best_tree = max(("CatBoost", "LightGBM", "XGBoost", "RandomForest"), key=lambda k: T[k].mean())
    print("   best tuned tree:", best_tree, ms(T[best_tree]), "| SBGA-F+CatBoost-tuned - best tree", fmt(paired(T["SBGA-F + CatBoost-tuned"], T[best_tree])))


def sec_causal():
    d = read(R2, "causal")
    if d.empty:
        return
    print("\n===== CAUSAL FLOW FEATURES =====")
    C = {m: g.set_index("seed")["macro_F1"] for m, g in d.groupby("model", sort=False)}
    CR = {m: g.set_index("seed")["min_class_recall"] for m, g in d.groupby("model", sort=False)}
    for m in C:
        line = f"   {m:18s} causal {ms(C[m]):12s} minrec {ms(CR[m]):12s}"
        if m in M:
            line += f" | primary {ms(M[m])} | causal-primary {fmt(paired(C[m], M[m]))}"
        print(line)
    for a in ("SBGA-F-Stack", "RNN+CatBoost", "NoGraph+CatBoost", "SBGA-F", "RNN", "SBGA-F_no_graph"):
        if a in C:
            print(f"   {a} - CatBoost (causal): {fmt(paired(C[a], C['CatBoost']))}")
    if "SBGA-F_no_graph" in C:
        print("   SBGA-F - no_graph:", fmt(paired(C["SBGA-F"], C["SBGA-F_no_graph"])), "| stack - nograph stack", fmt(paired(C["SBGA-F-Stack"], C["NoGraph+CatBoost"])))
        print("   SBGA-F - RNN:", fmt(paired(C["SBGA-F"], C["RNN"])), " SBGA-F - BiRAGNN:", fmt(paired(C["SBGA-F"], C["BiRAGNN"])))


def sec_interleaved():
    p = os.path.join(R2, "audit_interleaved_raw.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    print("\n===== INTERLEAVED STREAM =====")
    I = {m: g.set_index("seed")["macro_F1"] for m, g in d.groupby("model", sort=False)}
    for m in I:
        g = d[d.model == m]
        print(f"   {m:40s} {ms(I[m]):12s} minrec {g.min_recall.mean():.1f} | - CatBoost {fmt(paired(I[m], I['CatBoost']))}")
    print("   SBGA-F - no_graph", fmt(paired(I["SBGA-F"], I["SBGA-F_no_graph"])), "| SBGA-F - RNN", fmt(paired(I["SBGA-F"], I["RNN"])),
          "| SBGA-F - BiRAGNN", fmt(paired(I["SBGA-F"], I["BiRAGNN"])))
    print("   SBGA-F interleaved - capture order", fmt(paired(I["SBGA-F"], M["SBGA-F"])))
    print("   stack interleaved - capture order", fmt(paired(I["SBGA-F + CatBoost (stacked)"], M["SBGA-F-Stack"])))


def sec_stackctl():
    p = os.path.join(R2, "audit_stack_controls_raw.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    print("\n===== STACK CONTROLS (paired vs proposed and vs CatBoost) =====")
    S = {m: g.set_index("seed")["macro_F1"] for m, g in d.groupby("config", sort=False)}
    ref = S["SBGA-F + CatBoost, LR meta-learner (proposed)"]
    for m in S:
        g = d[d.config == m]
        print(f"   {m:66s} {ms(S[m]):12s} minrec {g.min_recall.mean():.1f} | proposed - row {fmt(paired(ref, S[m]))} | row - CatBoost {fmt(paired(S[m], S['CatBoost (single)']))}")
    mc = pd.read_csv(os.path.join(R2, "audit_mcnemar.csv"))
    print("   McNemar: seeds with p<0.05 and stack better:", int(((mc.mcnemar_p < 0.05) & (mc.stack_only_correct > mc.catboost_only_correct)).sum()),
          "| catboost better sig:", int(((mc.mcnemar_p < 0.05) & (mc.stack_only_correct < mc.catboost_only_correct)).sum()),
          "| bootstrap CI excludes 0 (positive):", int((mc.boot_lo > 0).sum()), "of", len(mc),
          f"| mean CI [{mc.boot_lo.mean():+.2f},{mc.boot_hi.mean():+.2f}]",
          f"| net events gained mean {(mc.stack_only_correct - mc.catboost_only_correct).mean():.0f}")


def sec_transfer():
    d = read(R2, "transfer")
    if d.empty:
        return
    print("\n===== GRAPH-SIZE REUSE (trained on 100 nodes) =====")
    T = {k: g.set_index("seed")["macro_F1"] for k, g in d.groupby(["model", "target", "frac"], sort=False)}
    for (m, k, f), s in T.items():
        base = T[(m, 100, 1.0)]
        print(f"   {m:22s} target {k:3d} frac {f:<5} {ms(s):12s} | vs own 100-node {fmt(paired(s, base))}")
    for k in (50, 8):
        for other in ("SBGA-F temporal-only", "BiRAGNN"):
            print(f"   target {k}: SBGA-F - {other}: {fmt(paired(T[('SBGA-F', k, 1.0)], T[(other, k, 1.0)]))}")
        print(f"   target {k}: stack - CatBoost: {fmt(paired(T[('SBGA-F-Stack', k, 1.0)], M['CatBoost']))}")
    b = os.path.join(R2, "transfer_budget_stats.csv")
    if os.path.exists(b):
        print(pd.read_csv(b).to_string(index=False))
    u = read(R2, "unseen")
    u["context"] = u["context"].fillna("n/a") if not u.empty else None
    if not u.empty:
        print("-- unseen devices (25 of 100 nodes withheld)")
        print(u.groupby(["slice", "model", "context"], sort=False).agg(n=("n", "mean"), classes=("n_classes", "mean"), macro_F1=("macro_F1", "mean"),
              sd=("macro_F1", "std"), accuracy=("accuracy", "mean"), seeds=("seed", "count")).round(2).to_string())
        U = {k: g.set_index("seed")["macro_F1"] for k, g in u.groupby(["slice", "model", "context"], sort=False)}
        for other, c in (("SBGA-F temporal-only", "rebuilt"), ("BiRAGNN", "rebuilt"), ("RNN", "rebuilt"), ("CatBoost", "n/a"), ("SBGA-F", "training graph (no update)")):
            print(f"   unseen nodes: SBGA-F(rebuilt) - {other}: {fmt(paired(U[('unseen nodes', 'SBGA-F', 'rebuilt')], U[('unseen nodes', other, c)]))}")


def sec_robust():
    d = read(R2, "robust")
    if d.empty:
        return
    print("\n===== ROBUSTNESS CONTROLS (mean over seeds, draw 0) =====")
    d0 = d[d.draw == 0]
    print(d0.pivot_table(index=["condition", "perturbation", "level"], columns="model", values="macro_F1", sort=False).round(1).to_string())
    print("seeds:", d.seed.nunique())
    R = {k: g.set_index("seed")["macro_F1"] for k, g in d0.groupby(["condition", "perturbation", "level", "model"], sort=False)}
    for cond in [("feature", "num_noise", 0.25), ("feature", "missing", 0.1), ("feature", "combined", 1.0), ("fgsm_from_SBGA-F", "fgsm", 0.1)]:
        for a, b in (("SBGA-F", "CatBoost-aug"), ("SBGA-F-Stack", "CatBoost-aug"), ("SBGA-F-Stack-aug", "CatBoost-aug"), ("SBGA-F-Stack-aug", "LightGBM-aug"), ("CatBoost-aug", "CatBoost")):
            if cond + (a,) in R and cond + (b,) in R:
                print(f"   {cond}: {a} - {b}: {fmt(paired(R[cond + (a,)], R[cond + (b,)]))}")
    for k, l in (("num_noise", 0.25), ("num_noise", 0.5), ("missing", 0.1), ("missing", 0.25), ("combined", 1.0)):
        for m in ("SBGA-F", "BiRAGNN", "SBGA-F-Stack"):
            a, b = ("feature+context", k, l, m), ("feature", k, l, m)
            if a in R:
                print(f"   degraded context - clean context, {k} {l}, {m}: {fmt(paired(R[a], R[b]))}")
    dr = d[(d.condition == "draws") | ((d.condition == "feature") & d.perturbation.isin(["num_noise", "missing"]) & d.level.isin([0.25, 0.1]))]
    dr = dr[((dr.perturbation == "num_noise") & (dr.level == 0.25)) | ((dr.perturbation == "missing") & (dr.level == 0.1))]
    dr = dr[dr.model.isin(["SBGA-F", "CatBoost", "SBGA-F-Stack"])]
    g = dr.groupby(["perturbation", "model", "seed"])["macro_F1"].agg(["mean", "std", "count"])
    print("-- repeated draws: within-seed sd over draws (mean over seeds) vs between-seed sd of the draw means")
    print(g.groupby(["perturbation", "model"]).agg(mean=("mean", "mean"), within_draw_sd=("std", "mean"), between_seed_sd=("mean", "std"), draws=("count", "first")).round(3).to_string())
    c = d[d.condition == "draws"].groupby("perturbation")[["cells_changed", "frac_changed", "rows_changed"]].agg(["mean", "min", "max"])
    print(c.to_string())


def sec_ton():
    d = read(R2, "ton")
    if d.empty:
        return
    print("\n===== TON_IoT =====")
    for f in ("ton_info.csv", "ton_stats.csv"):
        print(pd.read_csv(os.path.join(R2, f)).to_string(index=False))
    T = {m: g.set_index("seed")["macro_F1"] for m, g in d.groupby("model", sort=False)}
    for m in T:
        g = d[d.model == m]
        print(f"   {m:28s} {ms(T[m]):12s} minrec {ms(g.min_class_recall):12s} balacc {g.balanced_acc.mean():.2f} | SBGA-F - row {fmt(paired(T['SBGA-F'], T[m]))} | stack - row {fmt(paired(T['SBGA-F-Stack'], T[m]))}")
    t = read(R2, "ton_transfer")
    if not t.empty:
        TT = {k: g.set_index("seed")["macro_F1"] for k, g in t.groupby(["model", "target"], sort=False)}
        for (m, k), s in TT.items():
            line = f"   reuse {m:28s} target {k:3d} {ms(s):12s}"
            if (m, 50) in TT and k != 50:
                line += f" | vs own 50-node {fmt(paired(s, TT[(m, 50)]))}"
            print(line)
        for k in (25, 8):
            for o in ("BiRAGNN", "RNN", "SBGA-F temporal-only", "SBGA-F retrained on target"):
                if (o, k) in TT:
                    print(f"   target {k}: SBGA-F - {o}: {fmt(paired(TT[('SBGA-F', k)], TT[(o, k)]))}")


def sec_memory():
    p = os.path.join(R2, "memory.csv")
    if os.path.exists(p):
        print("\n===== MEMORY / CPU =====")
        print(pd.read_csv(p).round(4).to_string(index=False))


SECS = dict(ablation=sec_ablation, tuned=sec_tuned, causal=sec_causal, interleaved=sec_interleaved, stackctl=sec_stackctl,
            transfer=sec_transfer, robust=sec_robust, ton=sec_ton, memory=sec_memory)
if __name__ == "__main__":
    for s in (sys.argv[1:] or list(SECS)):
        SECS[s]()
