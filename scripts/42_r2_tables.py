"""Round-2 LaTeX tables (complete table environments) written to revision/tables_r2/.
Usage: python 42_r2_tables.py"""
import glob, json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
A = import_module("39_r2_analysis")
paired, read, R1, R2, M, MR = A.paired, A.read, A.R1, A.R2, A.M, A.MR
OUT = os.path.join(A.ROOT, "revision", "tables_r2")
os.makedirs(OUT, exist_ok=True)


def pm(s, d=2):
    return f"${s.mean():.{d}f}\\pm{s.std(ddof=1):.{d}f}$"


def dp(r, d=2):
    p = r["p"]
    return f"${r['diff']:+.{d}f}$ ({p:.3f})"


def table(name, caption, label, colspec, header, rows, size="\\footnotesize", tabcolsep="3pt", pos="t"):
    body = "\n".join(rows)
    s = (f"\\begin{{table}}[{pos}]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n{size}\n\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n"
         f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{colspec}@{{}}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n"
         f"\\end{{tabular*}}\n\\end{{table}}\n")
    open(os.path.join(OUT, name), "w", encoding="utf-8").write(s)
    print("wrote", name)


def t_ablation():
    d = read(R2, "ablation")
    names = [("no_graph", "Temporal encoder only"), ("no_bidirectional", "Incoming attention only"),
             ("out_only", "Outgoing attention only"), ("no_structural", "No structural descriptors"),
             ("dense_aggregation", "Dense aggregation"), ("k2", "$k=2$"), ("k4", "$k=4$"),
             ("k12", "$k=12$"), ("kall", "All neighbors"), ("TCN_encoder", "TCN encoder"),
             ("CE_loss", "Cross-entropy loss")]
    rows = [f"\\textbf{{Full model}} ($k=6$) & {pm(M['SBGA-F'])} & --- & {pm(M['SBGA-F-Stack'])} & --- & ${MR['SBGA-F-Stack'].mean():.1f}$ \\\\"]
    for arm, lab in names:
        a = d[(d.arm == arm) & (d.level == "standalone")].set_index("seed")
        s = d[(d.arm == arm) & (d.level == "stack")].set_index("seed")
        rows.append(f"{lab} & {pm(a.macro_F1)} & {dp(paired(a.macro_F1, M['SBGA-F']))} & {pm(s.macro_F1)} & {dp(paired(s.macro_F1, M['SBGA-F-Stack']))} & ${s.min_class_recall.mean():.1f}$ \\\\")
    table("tab_ablation.tex", "Component ablation over ten seeds for standalone SBGA-F and for the complete stack (each variant stacked with the CatBoost model of the same seed). Dense aggregation replaces learned attention by edge-weighted averaging, and ``all neighbors'' removes top-$k$ selection. The last column is the minimum per-class recall of the stack. $\\Delta$ is the paired macro-F1 difference from the full model in points, with the Wilcoxon $p$-value in parentheses.",
          "tab:ablation", "lccccc", "Variant & Standalone & $\\Delta$ ($p$) & Stack & $\\Delta$ ($p$) & Min. rec.", rows, tabcolsep="2pt")


def t_perclass():
    pc = pd.read_csv(os.path.join(R2, "audit_per_class_raw.csv"))
    g = pc.groupby(["model", "cls"], sort=False)[["precision", "recall", "f1"]].mean()
    sup = pc.groupby("cls", sort=False)["support"].first()
    rows = []
    for c in sup.index:
        cells = []
        for m in ("SBGA-F", "CatBoost", "SBGA-F-Stack"):
            cells += [f"{g.loc[(m, c), k]:.1f}" for k in ("precision", "recall", "f1")]
        rows.append(f"{c.replace('_', chr(92) + '_')} & {sup[c]:,}".replace(",", "{,}") + " & " + " & ".join(cells) + " \\\\")
    hdr = (" & & \\multicolumn{3}{c}{SBGA-F} & \\multicolumn{3}{c}{CatBoost} & \\multicolumn{3}{c}{SBGA-F-Stack} \\\\\n"
           "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}\nClass & Test & P & R & F1 & P & R & F1 & P & R & F1")
    table("tab_perclass.tex", "Per-class precision (P), recall (R), and F1 in \\% (mean over ten seeds) with the number of test events per class.",
          "tab:perclass", "lrccccccccc", hdr, rows, tabcolsep="2pt")


def t_gain():
    d = pd.read_csv(os.path.join(R2, "audit_stack_controls_raw.csv"))
    S = {m: g.set_index("seed") for m, g in d.groupby("config", sort=False)}
    ref = S["SBGA-F + CatBoost, LR meta-learner (proposed)"].macro_F1
    cb = S["CatBoost (single)"].macro_F1
    order = [("CatBoost (single)", "CatBoost"),
             ("CatBoost + LR meta-learner (CatBoost only)", "CatBoost + the same meta-learner (CatBoost probabilities only)"),
             ("CatBoost + CatBoost (second seed), stacked", "CatBoost + CatBoost trained with a second seed, stacked"),
             ("Temporal-only SBGA-F + CatBoost, stacked", "Temporal-only SBGA-F (no graph inputs) + CatBoost, stacked"),
             ("SBGA-F + CatBoost, LR meta-learner (proposed)", "\\textbf{SBGA-F-Stack} (logistic regression on probabilities)"),
             ("SBGA-F + CatBoost, LR with C=0.1", "SBGA-F + CatBoost, logistic regression with $C=0.1$"),
             ("SBGA-F + CatBoost, LR on log-probabilities", "SBGA-F + CatBoost, logistic regression on log-probabilities"),
             ("SBGA-F + CatBoost, MLP meta-learner", "SBGA-F + CatBoost, MLP meta-learner (32 hidden units)"),
             ("SBGA-F + CatBoost, LightGBM meta-learner", "SBGA-F + CatBoost, LightGBM meta-learner"),
             ("SBGA-F + CatBoost, unweighted average", "SBGA-F + CatBoost, unweighted probability average"),
             ("SBGA-F + CatBoost, weighted average (weight tuned on validation)", "SBGA-F + CatBoost, weighted average (weight chosen on validation)")]
    rows = []
    for k, lab in order:
        s = S[k]
        a = "---" if k == "CatBoost (single)" else dp(paired(s.macro_F1, cb))
        b = "---" if "proposed" in k else dp(paired(ref, s.macro_F1))
        rows.append(f"{lab} & {pm(s.macro_F1)} & ${s.min_recall.mean():.1f}$ & {a} & {b} \\\\")
    table("tab_gain.tex", "Decomposition of the stacking gain and alternative meta-learners (ten seeds). Row $-$ CatBoost and Stack $-$ row are paired macro-F1 differences in points (Wilcoxon $p$).",
          "tab:gain", "p{0.43\\textwidth}cccc", "Configuration & Macro-F1 & Min. rec. & Row $-$ CatBoost & Stack $-$ row", rows, tabcolsep="2pt")


def t_leakage():
    it = pd.read_csv(os.path.join(R2, "audit_interleaved_raw.csv"))
    ca = read(R2, "causal")
    ab = read(R2, "ablation")
    I = {m: g.set_index("seed").macro_F1 for m, g in it.groupby("model")}
    C = {m: g.set_index("seed").macro_F1 for m, g in ca.groupby("model")}
    ng = ab[(ab.arm == "no_graph") & (ab.level == "standalone")].set_index("seed").macro_F1
    ngs = ab[(ab.arm == "no_graph") & (ab.level == "stack")].set_index("seed").macro_F1
    sc = pd.read_csv(os.path.join(A.ROOT, "results", "rev2", "audit_stack_controls_raw.csv"))
    import exp_common as E
    from sklearn.metrics import f1_score
    base = E.load_base(); yv, yt = base.y[base.split_pos["val"]], base.y[base.split_pos["test"]]
    rc = {}
    for s in E.SEEDS10:
        meta = E.fit_meta([E.load_probs(s, "RNN", "val"), E.load_probs(s, "CatBoost", "val")], yv)
        rc[s] = f1_score(yt, meta.predict(np.concatenate([E.load_probs(s, "RNN", "test"), E.load_probs(s, "CatBoost", "test")], 1)), average="macro") * 100
    rc = pd.Series(rc)
    spec = [("CatBoost", M["CatBoost"], I["CatBoost"], C["CatBoost"]),
            ("RNN", M["RNN"], I["RNN"], C["RNN"]),
            ("BiRAGNN", M["BiRAGNN"], I["BiRAGNN"], C["BiRAGNN"]),
            ("Temporal-only SBGA-F", ng, I["SBGA-F_no_graph"], C["SBGA-F_no_graph"]),
            ("SBGA-F", M["SBGA-F"], I["SBGA-F"], C["SBGA-F"]),
            ("RNN + CatBoost, stacked", rc, I["RNN + CatBoost (stacked)"], C["RNN+CatBoost"]),
            ("Temporal-only SBGA-F + CatBoost, stacked", ngs, I["SBGA-F_no_graph + CatBoost (stacked)"], C["NoGraph+CatBoost"]),
            ("\\textbf{SBGA-F-Stack}", M["SBGA-F-Stack"], I["SBGA-F + CatBoost (stacked)"], C["SBGA-F-Stack"])]
    rows = []
    for lab, a, b, c in spec:
        rows.append(f"{lab} & {pm(a)} & {pm(b)} & {pm(c)} \\\\")
        if lab == "SBGA-F":
            rows.append("\\midrule")
    r = [paired(M["SBGA-F-Stack"], M["CatBoost"]), paired(I["SBGA-F + CatBoost (stacked)"], I["CatBoost"]), paired(C["SBGA-F-Stack"], C["CatBoost"])]
    rows.append("\\midrule")
    rows.append("SBGA-F-Stack $-$ CatBoost ($p$) & " + " & ".join(dp(x) for x in r) + " \\\\")
    r = [paired(M["SBGA-F"], ng), paired(I["SBGA-F"], I["SBGA-F_no_graph"]), paired(C["SBGA-F"], C["SBGA-F_no_graph"])]
    rows.append("SBGA-F $-$ temporal-only ($p$) & " + " & ".join(dp(x) for x in r) + " \\\\")
    table("tab_leakage.tex", "Leakage controls: test macro-F1 (\\%, mean $\\pm$ s.d. over ten seeds) under the primary capture-order protocol, the class-interleaved stream, and the causal flow-feature set. The last two rows give paired differences in points (Wilcoxon $p$).",
          "tab:leakage", "lccc", "Model & Capture order & Interleaved stream & Causal flow features", rows)
    print("RNN+CatBoost capture-order", rc.mean())


def t_robust():
    d = read(R2, "robust")
    d = d[d.draw == 0]
    cols = ["SBGA-F", "BiRAGNN", "CatBoost", "CatBoost-aug", "LightGBM-aug", "SBGA-F-Stack", "SBGA-F-Stack-aug"]
    spec = [("clean", "clean", 0.0, "Clean"), ("feature", "num_noise", 0.25, "Numeric noise $\\sigma=0.25$"), ("feature", "num_noise", 0.5, "Numeric noise $\\sigma=0.5$"),
            ("feature", "missing", 0.1, "Missing cells 10\\%"), ("feature", "missing", 0.25, "Missing cells 25\\%"),
            ("feature", "cat_corrupt", 0.1, "Categorical corruption 10\\%"), ("feature", "flow_dropout", 1.0, "Flow-exporter outage"),
            ("feature", "jitter", 0.5, "Timing/size jitter $s=0.5$"), ("feature", "combined", 1.0, "Combined"), None,
            ("feature+context", "missing", 0.1, "Missing 10\\%, degraded context"), ("feature+context", "missing", 0.25, "Missing 25\\%, degraded context"),
            ("feature+context", "num_noise", 0.5, "Noise $\\sigma=0.5$, degraded context"), ("feature+context", "combined", 1.0, "Combined, degraded context"),
            ("stale_graph", "stale", 0.1, "Stale context (first 10\\% of stream)"), ("stale_graph", "stale", 0.01, "Stale context (first 1\\% of stream)"), None,
            ("fgsm_from_SBGA-F", "fgsm", 0.05, "FGSM $\\epsilon=0.05$"), ("fgsm_from_SBGA-F", "fgsm", 0.1, "FGSM $\\epsilon=0.1$"), ("fgsm_from_SBGA-F", "fgsm", 0.2, "FGSM $\\epsilon=0.2$")]
    rows = []
    for sp in spec:
        if sp is None:
            rows.append("\\midrule"); continue
        c, k, l, lab = sp
        g = d[(d.condition == c) & (d.perturbation == k) & (np.isclose(d.level, l))].groupby("model").macro_F1.mean()
        vals = [g.get(m, np.nan) for m in cols]
        best = np.nanmax(vals)
        rows.append(lab + " & " + " & ".join(("\\textbf{%.1f}" % v if abs(v - best) < 0.05 else "%.1f" % v) for v in vals) + " \\\\")
    table("tab_robust2.tex", "Robustness controls: test macro-F1 (\\%, mean over ten seeds). ``-aug'' denotes training with noise and missing-cell augmentation; SBGA-F-Stack-aug stacks SBGA-F with the augmented CatBoost. The FGSM perturbation is computed against SBGA-F and transferred to the other models. The best value in each row is in bold.",
          "tab:robust2", "lccccccc", "Condition & SBGA-F & BiRAG. & CatB. & CatB.-aug & LGBM-aug & Stack & Stack-aug", rows, tabcolsep="2pt")


def t_ton():
    d = read(R2, "ton")
    T = {m: g.set_index("seed") for m, g in d.groupby("model", sort=False)}
    order = [("CatBoost", "CatBoost"), ("LightGBM", "LightGBM"), ("XGBoost", "XGBoost"), ("RandomForest", "Random Forest"), ("RNN", "RNN"),
             ("SBGA-F temporal-only", "Temporal-only SBGA-F"), ("GNN", "GNN"), ("BiRAGNN", "BiRAGNN"), ("SBGA-F", "\\textbf{SBGA-F}"), None,
             ("LightGBM + CatBoost", "LightGBM + CatBoost, stacked"), ("RNN + CatBoost", "RNN + CatBoost, stacked"),
             ("Temporal-only + CatBoost", "Temporal-only SBGA-F + CatBoost, stacked"), ("BiRAGNN + CatBoost", "BiRAGNN + CatBoost, stacked"),
             ("SBGA-F + CatBoost, average", "SBGA-F + CatBoost, probability average"), ("SBGA-F-Stack", "\\textbf{SBGA-F-Stack}")]
    rows = []
    for o in order:
        if o is None:
            rows.append("\\midrule"); continue
        k, lab = o
        s = T[k]
        a = "---" if k == "SBGA-F-Stack" else dp(paired(T["SBGA-F-Stack"].macro_F1, s.macro_F1))
        rows.append(f"{lab} & {pm(s.macro_F1)} & ${s.balanced_acc.mean():.1f}$ & ${s.min_class_recall.mean():.1f}$ & {a} \\\\")
    table("tab_ton.tex", "TON\\_IoT (ten classes, 50-node graph, ten seeds, hyperparameters unchanged from Edge-IIoTset): test macro-F1 (mean $\\pm$ s.d.), balanced accuracy, minimum per-class recall (\\%), and paired difference of SBGA-F-Stack minus the row in points (Wilcoxon $p$).",
          "tab:ton", "lcccc", "Model & Macro-F1 & Bal. acc. & Min. rec. & Stack $-$ row", rows)
    t = read(R2, "ton_transfer")
    TT = {k: g.set_index("seed").macro_F1 for k, g in t.groupby(["model", "target"], sort=False)}
    rows = []
    for m, lab in (("SBGA-F", "SBGA-F"), ("SBGA-F temporal-only", "Temporal-only SBGA-F"), ("BiRAGNN", "BiRAGNN"), ("RNN", "RNN")):
        rows.append(f"{lab} & {pm(TT[(m, 50)])} & {pm(TT[(m, 25)])} & {pm(TT[(m, 8)])} \\\\")
    rows.append(f"SBGA-F retrained on target & -- & {pm(TT[('SBGA-F retrained on target', 25)])} & {pm(TT[('SBGA-F retrained on target', 8)])} \\\\")
    table("tab_ton_reuse.tex", "TON\\_IoT: parameter reuse across graph sizes. Models are trained on the 50-node graph and applied to 25- and 8-node graphs after the graph context is recomputed (test macro-F1 in \\%, ten seeds).",
          "tab:tonreuse", "lccc", "Model & 50 nodes (training) & 25 nodes & 8 nodes", rows)


def t_raw():
    r = pd.read_csv(os.path.join(R2, "audit_raw_paired_scores.csv")).set_index("seed")
    mc = pd.read_csv(os.path.join(R2, "audit_mcnemar.csv")).set_index("seed")
    rows = []
    for s in r.index:
        rows.append(f"{s} & {r.loc[s, 'SBGA-F']:.2f} & {r.loc[s, 'CatBoost']:.2f} & {r.loc[s, 'SBGA-F-Stack']:.2f} & ${mc.loc[s, 'macro_F1_diff']:+.2f}$ & $[{mc.loc[s, 'boot_lo']:+.2f},{mc.loc[s, 'boot_hi']:+.2f}]$ & {int(mc.loc[s, 'stack_only_correct'])} / {int(mc.loc[s, 'catboost_only_correct'])} & {mc.loc[s, 'mcnemar_p']:.3f} \\\\")
    rows.append("\\midrule")
    rows.append(f"Mean & {r['SBGA-F'].mean():.2f} & {r['CatBoost'].mean():.2f} & {r['SBGA-F-Stack'].mean():.2f} & ${mc.macro_F1_diff.mean():+.2f}$ & $[{mc.boot_lo.mean():+.2f},{mc.boot_hi.mean():+.2f}]$ & {mc.stack_only_correct.mean():.0f} / {mc.catboost_only_correct.mean():.0f} & -- \\\\")
    table("tab_raw.tex", "Seed-level test macro-F1 (\\%), the difference SBGA-F-Stack minus CatBoost with its event-level bootstrap 95\\% interval, the number of test events that only the stack or only CatBoost classifies correctly, and the exact McNemar $p$-value.",
          "tab:raw", "lccccccc", "Seed & SBGA-F & CatBoost & Stack & Diff. & Bootstrap 95\\% & Only stack / only CatB. & McNemar $p$", rows, pos="h", tabcolsep="2pt")


def t_reuse():
    d = read(R2, "transfer")
    if d.empty:
        return
    T = {k: g.set_index("seed").macro_F1 for k, g in d.groupby(["model", "target", "frac"], sort=False)}
    rows = []
    for m, lab in (("SBGA-F", "SBGA-F"), ("SBGA-F temporal-only", "Temporal-only SBGA-F"), ("BiRAGNN", "BiRAGNN"),
                   ("Node-embedding LSTM", "Node-indexed LSTM"), ("SBGA-F-Stack", "SBGA-F-Stack")):
        sub = lambda x: f"${x.mean():.1f}_{{{x.std(ddof=1):.1f}}}$"
        rows.append(f"{lab} & {sub(T[(m, 100, 1.0)])} & " + " & ".join(sub(T[(m, k, f)]) for k in (50, 8) for f in (1.0, 0.1, 0.01)) + " \\\\")
    hdr = (" & 100 nodes & \\multicolumn{3}{c}{50-node target} & \\multicolumn{3}{c}{8-node target} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
           "Model & (training) & 100\\% & 10\\% & 1\\% & 100\\% & 10\\% & 1\\%")
    table("tab_reuse.tex", "Parameter reuse across graph sizes on Edge-IIoTset, second set of runs (test macro-F1 in \\%, mean over ten seeds with the standard deviation as subscript). Models are trained on the 100-node graph. For each target, the adjacency, descriptors, neighborhoods, and prototypes are recomputed from the stated fraction of the unlabeled target training stream (100\\% = 208{,}175 events).",
          "tab:reuse", "lccccccc", hdr, rows, tabcolsep="2pt")
    u = read(R2, "unseen")
    u["context"] = u["context"].fillna("n/a") if not u.empty else None
    if u.empty:
        return
    U = {k: g.set_index("seed") for k, g in u.groupby(["model", "context", "slice"], sort=False)}
    rows = []
    for m, c, lab in (("SBGA-F", "rebuilt", "SBGA-F, context rebuilt from unlabeled events"), ("SBGA-F", "training graph (no update)", "SBGA-F, training graph kept (no update)"),
                      ("SBGA-F temporal-only", "rebuilt", "Temporal-only SBGA-F"), ("BiRAGNN", "rebuilt", "BiRAGNN, context rebuilt"), ("RNN", "rebuilt", "RNN"), ("CatBoost", "n/a", "CatBoost")):
        a, b = U[(m, c, "unseen nodes")], U[(m, c, "seen nodes")]
        rows.append(f"{lab} & {pm(a.macro_F1)} & ${a.accuracy.mean():.1f}$ & {pm(b.macro_F1)} & ${b.accuracy.mean():.1f}$ \\\\")
    a = U[("SBGA-F", "rebuilt", "unseen nodes")]
    table("tab_unseen.tex", f"Unseen devices on Edge-IIoTset: 25 of the 100 nodes are withheld from training and validation (a different random set per seed, ten seeds). Macro-F1 is computed over the classes with at least 20 test events in the slice (on average {a.n.mean():,.0f} events and {a.n_classes.mean():.1f} classes at withheld nodes).".replace(",", "{,}", 1),
          "tab:unseen", "lcccc", " & \\multicolumn{2}{c}{Withheld nodes} & \\multicolumn{2}{c}{Training nodes} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nModel & Macro-F1 & Acc. & Macro-F1 & Acc.", rows)


def t_tuned():
    d = read(R2, "tuned")
    if d.empty:
        return
    T = {m: g.set_index("seed") for m, g in d.groupby("model", sort=False)}
    rows = []
    for m, lab in (("CatBoost", "CatBoost"), ("LightGBM", "LightGBM"), ("XGBoost", "XGBoost"), ("RandomForest", "Random Forest"), ("BiRAGNN", "BiRAGNN"), ("RNN", "RNN")):
        rows.append(f"{lab} & {pm(M[m])} & {pm(T[m].macro_F1)} & {dp(paired(T[m].macro_F1, M[m]))} & {dp(paired(M['SBGA-F'], T[m].macro_F1))} \\\\")
    rows.append("\\midrule")
    cb = T["CatBoost"].macro_F1
    for m, lab in (("SBGA-F + CatBoost-tuned", "SBGA-F + tuned CatBoost"), ("RNN-tuned + CatBoost-tuned", "Tuned RNN + tuned CatBoost"),
                   ("BiRAGNN-tuned + CatBoost-tuned", "Tuned BiRAGNN + tuned CatBoost")):
        rows.append(f"{lab} & -- & {pm(T[m].macro_F1)} & \\multicolumn{{2}}{{c}}{{$-$ tuned CatBoost: {dp(paired(T[m].macro_F1, cb))}}} \\\\")
    table("tab_tuned.tex", "Equal tuning budget (20 TPE trials on validation macro-F1 per model): test macro-F1 (\\%, ten seeds) with the fixed and the tuned configuration, the paired change, and the paired difference of SBGA-F (96.46) minus the tuned model, in points (Wilcoxon $p$). The last three rows are stacks.",
          "tab:tuned", "lcccc", "Model & Fixed & Tuned & Tuned $-$ fixed & SBGA-F $-$ tuned", rows)


if __name__ == "__main__":
    for f in (t_ablation, t_perclass, t_gain, t_leakage, t_robust, t_ton, t_raw, t_reuse, t_tuned):
        try:
            f()
        except Exception as e:
            print("FAILED", f.__name__, repr(e))
