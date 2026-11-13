"""
Aggregates the revision experiments into LaTeX table fragments, figures and a plain-text summary.
Reads results/rev/*.csv and results/probs/*.npy; writes revision/tables/*.tex, revision/figures/*.pdf,
revision/summary.txt.  Usage: python 22_rev_analysis.py
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "rev")
PROB = os.path.join(ROOT, "results", "probs")
OUT = os.path.join(ROOT, "revision")
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")
for d in (TAB, FIG):
    os.makedirs(d, exist_ok=True)

SUMMARY = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    SUMMARY.append(s)


def read_all(prefix):
    fs = sorted(glob.glob(os.path.join(RES, f"{prefix}_*.csv")))
    fs = [f for f in fs if os.path.basename(f)[len(prefix) + 1:-4].isdigit()]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def pm(m, s, d=2):
    return f"{m:.{d}f}\\pm{s:.{d}f}"


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired(a, b):
    """paired statistics for arrays aligned by seed (a - b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    res = dict(n=n, mean_diff=d.mean(), n_pos=int((d > 0).sum()))
    if n > 1:
        se = d.std(ddof=1) / np.sqrt(n)
        t = stats.t.ppf(0.975, n - 1)
        res["ci_lo"], res["ci_hi"] = d.mean() - t * se, d.mean() + t * se
    else:
        res["ci_lo"] = res["ci_hi"] = float("nan")
    try:
        res["p"] = stats.wilcoxon(a, b, zero_method="wilcox", method="auto").pvalue if np.any(d != 0) else 1.0
    except Exception:
        res["p"] = float("nan")
    # matched-pairs rank-biserial correlation
    nz = d[d != 0]
    if len(nz):
        ranks = stats.rankdata(np.abs(nz))
        res["rbc"] = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
    else:
        res["rbc"] = 0.0
    return res


def fmt_p(p):
    if p is None or np.isnan(p):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def write(name, text):
    with open(os.path.join(TAB, name), "w", encoding="utf8") as f:
        f.write(text)


# ----------------------------------------------------------------------------------
# A. main benchmark
# ----------------------------------------------------------------------------------
def main_table():
    df = read_all("main")
    if df.empty:
        return None
    seeds = sorted(df.seed.unique())
    say(f"[main] seeds available: {seeds}")
    g = df.groupby("model")
    agg = g.agg(f1=("macro_F1", "mean"), sd=("macro_F1", "std"), minrec=("min_class_recall", "mean"),
                bal=("balanced_acc", "mean"), infer=("infer_ms_per_sample", "mean"),
                params=("n_params", "mean"), cat=("category", "first"), n=("seed", "count")).reset_index()
    agg = agg.sort_values("f1", ascending=False)
    piv = df.pivot(index="seed", columns="model", values="macro_F1")
    rows = []
    sb = piv["SBGA-F"]
    others = [m for m in piv.columns if m not in ("SBGA-F", "SBGA-F-Stack")]
    ps = {m: paired(sb, piv[m]) for m in others}
    adj = dict(zip(others, holm([ps[m]["p"] for m in others])))
    say("[main] SBGA-F vs each benchmark (paired, seeds=%d): mean diff, Wilcoxon p, Holm p" % len(piv))
    for m in sorted(others, key=lambda k: -piv[k].mean()):
        say(f"   {m:20s} diff={ps[m]['mean_diff']:+.2f}  CI=[{ps[m]['ci_lo']:+.2f},{ps[m]['ci_hi']:+.2f}]  p={ps[m]['p']:.4f}  holm={adj[m]:.4f}")
    cat_name = {"classical": "Classical", "tree": "Tree ensemble", "deep": "Deep", "graph": "Graph",
                "proposed": "Proposed", "proposed_stack": "Proposed"}
    disp = {"LogisticRegression": "Logistic Regression", "LinearSVM_SGD": "Linear SVM", "kNN": "$k$-NN",
            "RandomForest": "Random Forest", "FT-Transformer": "FT-Transformer", "SBGA-F-Stack": "SBGA-F-Stack"}
    lines = []
    for _, r in agg.iterrows():
        name = disp.get(r["model"], r["model"])
        bold = r["cat"].startswith("proposed")
        f1s = pm(r["f1"], r["sd"])
        row_vs = ""
        if r["model"] in ps:
            row_vs = f"{ps[r['model']]['mean_diff']:+.2f}"
        elif r["model"] == "SBGA-F":
            row_vs = "---"
        else:
            row_vs = f"{paired(sb, piv[r['model']])['mean_diff']:+.2f}"
        par = "--" if np.isnan(r["params"]) or r["params"] == 0 or r["cat"] in ("classical", "tree") else f"{int(r['params']):,}".replace(",", "{,}")
        inf = f"{r['infer']:.3f}" if r["infer"] >= 0.001 else "$<$0.001"
        if r["model"] == "SBGA-F-Stack":
            inf, par = "--", "--"
        f1cell = f"$\\mathbf{{{f1s}}}$" if bold else f"${f1s}$"
        nm = f"\\textbf{{{name}}}" if bold else name
        lines.append(f"{cat_name[r['cat']]} & {nm} & {f1cell} & {r['minrec']:.1f} & {row_vs} & {par} \\\\")
    write("main_benchmark_rows.tex", "\n".join(lines) + "\n")
    say("[main] table:")
    for _, r in agg.iterrows():
        say(f"   {r['model']:20s} F1={r['f1']:.2f}+-{r['sd']:.2f} minRec={r['minrec']:.1f} infer={r['infer']:.4f} params={r['params']}")
    return df, piv


# ----------------------------------------------------------------------------------
# B. stacking controls, per-class, paired uncertainty (from stored probabilities)
# ----------------------------------------------------------------------------------
def load_p(seed, name, split):
    return np.load(os.path.join(PROB, f"{seed}_{name}_{split}.npy"))


def fast_f1(y, pred, k=15):
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k).astype(float)
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return f1.mean() * 100, rec.min() * 100, prec, rec, f1


def get_labels():
    sys.path.insert(0, os.path.dirname(__file__))
    import exp_common as E
    base = E.load_base()
    return base.y[base.split_pos["val"]], base.y[base.split_pos["test"]], list(base.classes)


def stack_controls():
    seeds = sorted({int(os.path.basename(f).split("_")[0]) for f in glob.glob(os.path.join(PROB, "*_SBGA-F-Stack_test.npy"))})
    if not seeds:
        return
    y_va, y_te, classes = get_labels()
    say(f"[stack] seeds with stored probabilities: {seeds}")
    combos = [
        ("SBGA-F-Stack (proposed)", "stack", ["SBGA-F", "CatBoost"]),
        ("SBGA-F + CatBoost, probability average", "avg", ["SBGA-F", "CatBoost"]),
        ("SBGA-F + CatBoost, val-weighted average", "wavg", ["SBGA-F", "CatBoost"]),
        ("BiRAGNN + CatBoost, stacked", "stack", ["BiRAGNN", "CatBoost"]),
        ("BiRAGNN + CatBoost, average", "avg", ["BiRAGNN", "CatBoost"]),
        ("RNN (LSTM) + CatBoost, stacked", "stack", ["RNN", "CatBoost"]),
        ("RNN (LSTM) + CatBoost, average", "avg", ["RNN", "CatBoost"]),
        ("GNN + CatBoost, stacked", "stack", ["GNN", "CatBoost"]),
        ("TGNN + CatBoost, stacked", "stack", ["TGNN", "CatBoost"]),
        ("LightGBM + CatBoost, stacked", "stack", ["LightGBM", "CatBoost"]),
        ("SBGA-F + LightGBM, stacked", "stack", ["SBGA-F", "LightGBM"]),
        ("SBGA-F + BiRAGNN + CatBoost, stacked", "stack", ["SBGA-F", "BiRAGNN", "CatBoost"]),
        ("CatBoost (single)", "single", ["CatBoost"]),
        ("SBGA-F (single)", "single", ["SBGA-F"]),
        ("BiRAGNN (single)", "single", ["BiRAGNN"]),
        ("RNN (single)", "single", ["RNN"]),
    ]
    res = {c[0]: [] for c in combos}
    minrec = {c[0]: [] for c in combos}
    per_class = {}
    preds_cache = {}
    for s in seeds:
        P = {n: (load_p(s, n, "val"), load_p(s, n, "test")) for n in ["SBGA-F", "CatBoost", "LightGBM", "BiRAGNN", "RNN", "GNN", "TGNN"]}
        for label, kind, members in combos:
            if kind == "single":
                pt = P[members[0]][1]
            elif kind == "avg":
                pt = np.mean([P[m][1] for m in members], axis=0)
            elif kind == "wavg":
                best, bw = -1, 0.5
                for w in np.linspace(0, 1, 21):
                    f = fast_f1(y_va, (w * P[members[0]][0] + (1 - w) * P[members[1]][0]).argmax(1))[0]
                    if f > best:
                        best, bw = f, w
                pt = bw * P[members[0]][1] + (1 - bw) * P[members[1]][1]
            else:
                meta = LogisticRegression(max_iter=2000).fit(np.concatenate([P[m][0] for m in members], axis=1), y_va)
                pt = meta.predict_proba(np.concatenate([P[m][1] for m in members], axis=1))
            pred = pt.argmax(1)
            f1, mr, prec, rec, f1c = fast_f1(y_te, pred)
            res[label].append(f1)
            minrec[label].append(mr)
            if label in ("SBGA-F-Stack (proposed)", "CatBoost (single)", "SBGA-F (single)"):
                per_class.setdefault(label, []).append((prec * 100, rec * 100, f1c * 100))
            if label in ("SBGA-F-Stack (proposed)", "CatBoost (single)") and s == seeds[0]:
                preds_cache[label] = pred
    ref = np.array(res["SBGA-F-Stack (proposed)"])
    lines = []
    say("[stack] controls (mean+-sd macro-F1 over %d seeds; paired diff of proposed stack minus row)" % len(seeds))
    for label, kind, members in combos:
        v = np.array(res[label])
        pr = paired(ref, v) if label != "SBGA-F-Stack (proposed)" else None
        say(f"   {label:46s} {v.mean():.2f}+-{v.std(ddof=1):.2f}  minRec={np.mean(minrec[label]):.1f}" +
            (f"  stack-row diff={pr['mean_diff']:+.2f} [{pr['ci_lo']:+.2f},{pr['ci_hi']:+.2f}] p={pr['p']:.4f}" if pr else ""))
        diff = "---" if pr is None else f"${pr['mean_diff']:+.2f}$ ({fmt_p(pr['p'])})"
        nm = f"\\textbf{{{label}}}" if pr is None else label
        lines.append(f"{nm} & ${pm(v.mean(), v.std(ddof=1))}$ & {np.mean(minrec[label]):.1f} & {diff} \\\\")
    write("stack_controls_rows.tex", "\n".join(lines) + "\n")

    # per-class table
    pc = {k: np.mean([np.stack(x) for x in v], axis=0) for k, v in per_class.items()}  # [3, 15]
    rows = []
    order = np.argsort([-pc["CatBoost (single)"][2][i] for i in range(15)])
    for i in range(15):
        c = classes[i].replace("_", "\\_")
        rows.append(f"{c} & {pc['SBGA-F (single)'][1][i]:.1f} & {pc['CatBoost (single)'][1][i]:.1f} & {pc['SBGA-F-Stack (proposed)'][1][i]:.1f}"
                    f" & {pc['SBGA-F (single)'][2][i]:.1f} & {pc['CatBoost (single)'][2][i]:.1f} & {pc['SBGA-F-Stack (proposed)'][2][i]:.1f} \\\\")
    write("per_class_rows.tex", "\n".join(rows) + "\n")
    say("[per-class] mean over seeds, recall / F1 (SBGA-F, CatBoost, Stack)")
    for i in range(15):
        say(f"   {classes[i]:22s} rec {pc['SBGA-F (single)'][1][i]:5.1f} {pc['CatBoost (single)'][1][i]:5.1f} {pc['SBGA-F-Stack (proposed)'][1][i]:5.1f} | F1 {pc['SBGA-F (single)'][2][i]:5.1f} {pc['CatBoost (single)'][2][i]:5.1f} {pc['SBGA-F-Stack (proposed)'][2][i]:5.1f}")

    # event-level paired bootstrap (all seeds): CI of macro-F1 difference
    rng = np.random.default_rng(0)
    out = {}
    for pair in [("SBGA-F-Stack (proposed)", "CatBoost (single)"), ("SBGA-F-Stack (proposed)", "SBGA-F (single)")]:
        lo_list, hi_list, mean_list = [], [], []
        for s in seeds:
            P = {n: (load_p(s, n, "val"), load_p(s, n, "test")) for n in ["SBGA-F", "CatBoost"]}
            meta = LogisticRegression(max_iter=2000).fit(np.concatenate([P["SBGA-F"][0], P["CatBoost"][0]], axis=1), y_va)
            ps_ = meta.predict_proba(np.concatenate([P["SBGA-F"][1], P["CatBoost"][1]], axis=1)).argmax(1)
            pb = (P["CatBoost"][1] if "CatBoost" in pair[1] else P["SBGA-F"][1]).argmax(1)
            diffs = []
            for _ in range(300):
                idx = rng.integers(0, len(y_te), len(y_te))
                diffs.append(fast_f1(y_te[idx], ps_[idx])[0] - fast_f1(y_te[idx], pb[idx])[0])
            lo_list.append(np.percentile(diffs, 2.5)); hi_list.append(np.percentile(diffs, 97.5)); mean_list.append(np.mean(diffs))
        out[pair] = (np.mean(mean_list), np.mean(lo_list), np.mean(hi_list), np.mean(np.array(lo_list) > 0))
        say(f"[bootstrap] {pair[0]} - {pair[1]}: mean diff {out[pair][0]:+.2f}, mean 95% event-bootstrap CI [{out[pair][1]:+.2f},{out[pair][2]:+.2f}], "
            f"fraction of seeds with CI above 0: {out[pair][3]:.2f}")
    return res


# ----------------------------------------------------------------------------------
# C. perturbations
# ----------------------------------------------------------------------------------
def perturbation_tables():
    df = read_all("perturb")
    if df.empty:
        return
    seeds = sorted(df.seed.unique())
    say(f"[perturb] seeds: {seeds}")
    g = df.groupby(["model", "family", "perturbation", "level"]).agg(f1=("macro_F1", "mean"), sd=("macro_F1", "std"),
                                                                      mr=("min_recall", "mean")).reset_index()
    g.to_csv(os.path.join(TAB, "perturbation_summary.csv"), index=False)
    for (fam, pert), sub in g.groupby(["family", "perturbation"]):
        for lvl, s2 in sub.groupby("level"):
            say(f"   {fam:8s} {pert:12s} {lvl:5.2f} " + "  ".join(f"{r.model}={r.f1:.1f}" for r in s2.itertuples()))
    models = ["SBGA-F-Stack", "SBGA-F", "CatBoost", "LightGBM", "XGBoost", "RandomForest", "BiRAGNN", "RNN"]
    sel = [("feature", "clean", 0.0, "Clean"), ("feature", "num_noise", 0.25, "Numeric noise, $\\sigma{=}0.25$"),
           ("feature", "num_noise", 1.0, "Numeric noise, $\\sigma{=}1.0$"),
           ("feature", "cat_corrupt", 0.10, "Categorical corruption, 10\\%"),
           ("feature", "missing", 0.10, "Missing values, 10\\%"), ("feature", "missing", 0.50, "Missing values, 50\\%"),
           ("feature", "flow_dropout", 1.0, "Flow-exporter outage"), ("feature", "jitter", 0.5, "Timing/size jitter, $s{=}0.5$"),
           ("feature", "combined", 1.0, "Combined (mild)"),
           ("topology", "edge_drop", 0.3, "Edge dropout, 30\\%"), ("topology", "weight_noise", 0.5, "Edge-weight noise"),
           ("topology", "rewire", 0.3, "Edge rewiring, 30\\%")]
    rows = []
    for fam, pert, lvl, label in sel:
        cells = []
        for m in models:
            r = g[(g.model == m) & (g.family == fam) & (g.perturbation == pert) & (np.isclose(g.level, lvl))]
            cells.append(f"{r.f1.iloc[0]:.1f}" if len(r) else "--")
        best = [float(c) for c in cells if c != "--"]
        if best:
            mx = max(best)
            cells = [f"\\textbf{{{c}}}" if c != "--" and float(c) == mx else c for c in cells]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    write("perturbation_rows.tex", "\n".join(rows) + "\n")
    return g


# ----------------------------------------------------------------------------------
# D. ablation, transfer, graph controls
# ----------------------------------------------------------------------------------
def ablation_table(main):
    df = read_all("ablation")
    if df.empty or main is None:
        return
    full = main[0][main[0].model == "SBGA-F"].set_index("seed")
    seeds = sorted(df.seed.unique())
    seeds = [s for s in seeds if s in full.index]
    say(f"[ablation] seeds: {seeds}")
    rows = []
    name = {"no_bidirectional": "Unidirectional (in-neighbours only)", "no_structural": "No structural descriptors",
            "dense_aggregation": "Dense fixed-weight aggregation", "TCN_encoder": "TCN instead of LSTM+attention",
            "CE_loss": "Cross-entropy instead of focal", "no_graph": "No graph branch (temporal encoder only)"}
    f_ref = full.loc[seeds, "macro_F1"].values
    mr_ref = full.loc[seeds, "min_class_recall"].values
    rows.append(f"\\textbf{{Full SBGA-F}} & $\\mathbf{{{pm(f_ref.mean(), f_ref.std(ddof=1))}}}$ & ${pm(mr_ref.mean(), mr_ref.std(ddof=1), 1)}$ & --- \\\\")
    say(f"   Full  {f_ref.mean():.2f}+-{f_ref.std(ddof=1):.2f}")
    for arm in ["no_bidirectional", "no_structural", "CE_loss", "dense_aggregation", "TCN_encoder", "no_graph"]:
        sub = df[df.arm == arm].set_index("seed").loc[seeds]
        v = sub["macro_F1"].values
        pr = paired(v, f_ref)
        say(f"   {arm:18s} {v.mean():.2f}+-{v.std(ddof=1):.2f}  diff vs full {pr['mean_diff']:+.2f} [{pr['ci_lo']:+.2f},{pr['ci_hi']:+.2f}] p={pr['p']:.3f}")
        rows.append(f"{name[arm]} & ${pm(v.mean(), v.std(ddof=1))}$ & ${pm(sub['min_class_recall'].mean(), sub['min_class_recall'].std(ddof=1), 1)}$"
                    f" & ${pr['mean_diff']:+.2f}$ ({fmt_p(pr['p'])}) \\\\")
    write("ablation_rows.tex", "\n".join(rows) + "\n")


def transfer_table():
    df = read_all("transfer")
    tw = read_all("transfer_tw")
    if df.empty:
        return
    seeds = sorted(df.seed.unique())
    say(f"[transfer] seeds: {seeds}")
    if not tw.empty:
        df = pd.concat([df, tw], ignore_index=True)
    piv = df.pivot_table(index=["model", "target"], columns="seed", values="macro_F1")
    order = ["SBGA-F", "SBGA-F -struct", "SBGA-F -bidir", "SBGA-F (traffic-weighted descriptors)", "SBGA-F no graph",
             "BiRAGNN", "TGNN", "GNN", "RNN", "SBGA-F retrained on target"]
    rows = []
    for m in order:
        cells = []
        for k in (100, 50, 8):
            if (m, k) in piv.index:
                v = piv.loc[(m, k)].dropna().values
                cells.append(f"${pm(v.mean(), v.std(ddof=1))}$")
            else:
                cells.append("--")
        if all(c == "--" for c in cells):
            continue
        rows.append(f"{m.replace('-struct', '$-$structural').replace('-bidir', '$-$bidirectional')} & " + " & ".join(cells) + " \\\\")
        say(f"   {m:42s} " + " | ".join(c for c in cells))
    write("transfer_rows.tex", "\n".join(rows) + "\n")
    for k in (50, 8):
        for a, b in [("SBGA-F -struct", "SBGA-F"), ("SBGA-F -bidir", "SBGA-F"), ("SBGA-F (traffic-weighted descriptors)", "SBGA-F"),
                     ("SBGA-F", "SBGA-F retrained on target"), ("SBGA-F", "BiRAGNN"), ("SBGA-F", "RNN"), ("SBGA-F no graph", "SBGA-F")]:
            if (a, k) in piv.index and (b, k) in piv.index:
                common = piv.loc[(a, k)].dropna().index.intersection(piv.loc[(b, k)].dropna().index)
                if len(common) > 2:
                    pr = paired(piv.loc[(a, k)][common].values, piv.loc[(b, k)][common].values)
                    say(f"   target {k}: {a} - {b}: {pr['mean_diff']:+.2f} [{pr['ci_lo']:+.2f},{pr['ci_hi']:+.2f}] p={pr['p']:.3f} (n={pr['n']})")
    # transfer gap = (target F1) - (100-node F1), per model
    for m in order:
        for k in (50, 8):
            if (m, k) in piv.index and (m, 100) in piv.index:
                c = piv.loc[(m, k)].dropna().index.intersection(piv.loc[(m, 100)].dropna().index)
                if len(c):
                    say(f"   gap {m} 100->{k}: {np.mean(piv.loc[(m, k)][c].values - piv.loc[(m, 100)][c].values):+.2f}")
    return piv


def graphctl_table(main):
    df = read_all("graphctl")
    if df.empty:
        return
    seeds = sorted(df.seed.unique())
    say(f"[graph controls] seeds: {seeds}")
    full = main[0] if main else None
    rows = []
    labels = {("original", "transition"): "Event-transition (proposed)", ("original", "block_shuffle"): "Transitions, 200-event blocks shuffled",
              ("original", "shuffle"): "Transitions, event order shuffled", ("original", "link"): "Observed communication links (src$\\to$dst)",
              ("original", "rewired"): "Degree-preserving rewiring", ("interleaved", "transition"): "Interleaved stream (windows and graph)"}
    out = []
    for key in [("original", "transition"), ("original", "block_shuffle"), ("original", "shuffle"), ("original", "link"),
                ("original", "rewired"), ("interleaved", "transition")]:
        cells = []
        for m in ["SBGA-F", "BiRAGNN", "RNN", "SBGA-F_no_graph"]:
            if key == ("original", "transition"):
                if full is None:
                    cells.append("--"); continue
                sub = full[(full.model == m) & (full.seed.isin(seeds))]
                if m == "SBGA-F_no_graph":
                    ab = read_all("ablation")
                    sub = ab[(ab.arm == "no_graph") & (ab.seed.isin(seeds))] if not ab.empty else pd.DataFrame()
                v = sub["macro_F1"].values if len(sub) else np.array([])
            else:
                sub = df[(df.stream == key[0]) & (df.graph == key[1]) & (df.model == m)]
                v = sub["macro_F1"].values
            cells.append(f"${pm(v.mean(), v.std(ddof=1))}$" if len(v) > 1 else "--")
        out.append(f"{labels[key]} & " + " & ".join(cells) + " \\\\")
        say(f"   {str(key):32s} " + " | ".join(cells))
    write("graph_controls_rows.tex", "\n".join(out) + "\n")


def runtime_table():
    path = os.path.join(RES, "runtime.csv")
    if not os.path.exists(path):
        return
    r = pd.read_csv(path)
    lines = []
    for _, x in r.iterrows():
        ms = "--" if np.isnan(x["ms_per_event"]) else f"{x['ms_per_event']:.4f}"
        nm = x["stage"].replace("END-TO-END", "End-to-end")
        tot = "--" if ("single-event" in nm or "CPU, 8 threads" in nm) else f"{x['total_s']:.3f}"
        nm = nm.replace("SBGA-F forward (CPU, 8 threads, batch 1024)", "SBGA-F forward (CPU, 8 threads, batch 1024; 10{,}000 events)")
        lines.append(f"{nm} & {tot} & {ms} \\\\")
    write("runtime_rows.tex", "\n".join(lines) + "\n")


def main():
    main_res = main_table()
    stack_controls()
    perturbation_tables()
    ablation_table(main_res)
    transfer_table()
    graphctl_table(main_res)
    runtime_table()
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(SUMMARY))


if __name__ == "__main__":
    main()
