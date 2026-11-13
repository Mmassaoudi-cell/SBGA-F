"""Result figures for the revised manuscript (large labels, single-column width ~5.2 in)."""
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "rev")
FIG = os.path.join(ROOT, "revision", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "savefig.bbox": "tight", "pdf.fonttype": 42})
COL = {"SBGA-F-Stack": "#B22222", "SBGA-F": "#D55E00", "CatBoost": "#009E73", "LightGBM": "#66C2A5",
       "XGBoost": "#8DA0CB", "RandomForest": "#A6D854", "BiRAGNN": "#0072B2", "RNN": "#999999"}
MARK = {"SBGA-F-Stack": "s", "SBGA-F": "o", "CatBoost": "^", "LightGBM": "v", "XGBoost": "D", "RandomForest": "P",
        "BiRAGNN": "X", "RNN": "*"}


def read_all(prefix):
    fs = [f for f in sorted(glob.glob(os.path.join(RES, f"{prefix}_*.csv"))) if os.path.basename(f)[len(prefix) + 1:-4].isdigit()]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def fig_main():
    df = read_all("main")
    agg = df.groupby(["model", "category"])["macro_F1"].agg(["mean", "std"]).reset_index().sort_values("mean")
    cmap = {"classical": "#999999", "tree": "#009E73", "deep": "#56B4E9", "graph": "#CC79A7",
            "proposed": "#D55E00", "proposed_stack": "#B22222"}
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.barh(agg["model"], agg["mean"], xerr=agg["std"], color=[cmap[c] for c in agg["category"]], capsize=2.5, height=0.7)
    for i, (m, s) in enumerate(zip(agg["mean"], agg["std"])):
        ax.text(max(m, 0) + s + 0.15 if m > 84 else 84.3, i, f"{m:.2f}", va="center", fontsize=8)
    ax.set_xlim(84, 100.8)
    ax.set_xlabel("Test macro-F1 (%), mean $\\pm$ s.d. over 10 seeds")
    names = {"classical": "Classical", "tree": "Tree ensemble", "deep": "Deep", "graph": "Graph",
             "proposed": "SBGA-F", "proposed_stack": "SBGA-F-Stack"}
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.values()], [names[k] for k in cmap], loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=3, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(os.path.join(FIG, "fig_main_benchmark.pdf"))
    plt.close(fig)


def fig_perturb():
    df = read_all("perturb")
    if df.empty:
        return
    g = df.groupby(["model", "perturbation", "level"])["macro_F1"].agg(["mean", "std"]).reset_index()
    clean = g[(g.perturbation == "clean")].set_index("model")["mean"]
    panels = [("num_noise", "Numeric noise ($\\sigma$)", [0.25, 0.5, 1.0], (0, 100)),
              ("missing", "Missing cells (fraction)", [0.1, 0.25, 0.5], (0, 100)),
              ("cat_corrupt", "Categorical (fraction)", [0.05, 0.1, 0.25], (82, 100)),
              ("jitter", "Timing/size jitter ($s$)", [0.25, 0.5], (82, 100)),
              ("edge_drop", "Edge dropout (fraction)", [0.1, 0.3, 0.5], (90, 100)),
              ("rewire", "Edge rewiring (fraction)", [0.1, 0.3], (85, 100))]
    models = ["SBGA-F-Stack", "SBGA-F", "CatBoost", "LightGBM", "XGBoost", "BiRAGNN", "RNN"]
    fig, axes = plt.subplots(2, 3, figsize=(5.4, 4.2))
    for ax, (pert, title, levels, ylim) in zip(axes.ravel(), panels):
        for m in models:
            sub = g[(g.model == m) & (g.perturbation == pert)].sort_values("level")
            if sub.empty:
                continue
            xs = [0] + list(sub.level)
            ys = [clean[m]] + list(sub["mean"])
            es = [0] + list(sub["std"])
            ax.errorbar(xs, ys, yerr=es, color=COL[m], marker=MARK[m], ms=3.5, lw=1.2, capsize=1.5, label=m)
        ax.set_title(title, fontsize=7.5)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xticks([0] + levels)
        ax.set_xticklabels(["0"] + [f"{l:g}" for l in levels], fontsize=7)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1 (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_perturbation.pdf"))
    plt.close(fig)


def fig_transfer():
    df = read_all("transfer")
    tw = read_all("transfer_tw")
    if df.empty:
        return
    if not tw.empty:
        df = pd.concat([df, tw], ignore_index=True)
    order = [("SBGA-F", "SBGA-F"), ("SBGA-F -struct", "$-$struct."), ("SBGA-F -bidir", "$-$bidir."),
             ("SBGA-F (traffic-weighted descriptors)", "traffic-\nweighted"), ("BiRAGNN", "BiRAGNN"), ("RNN", "RNN")]
    order = [o for o in order if o[0] in set(df.model)]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    w = 0.26
    colors = {100: "#BBBBBB", 50: "#0072B2", 8: "#D55E00"}
    for j, tgt in enumerate([100, 50, 8]):
        mu, sd = [], []
        for m, _ in order:
            v = df[(df.model == m) & (df.target == tgt)]["macro_F1"].values
            mu.append(v.mean() if len(v) else np.nan)
            sd.append(v.std(ddof=1) if len(v) > 1 else 0)
        ax.bar(np.arange(len(order)) + (j - 1) * w, mu, w, yerr=sd, capsize=2, color=colors[tgt],
               label=f"{tgt} nodes" + (" (training)" if tgt == 100 else ""))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([o[1] for o in order], fontsize=7.5)
    lo = np.nanmin([df[df.model == m].macro_F1.mean() for m, _ in order]) - 3
    ax.set_ylim(91.5, 98.5)
    ax.set_ylabel("Macro-F1 (%)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(FIG, "fig_transfer.pdf"))
    plt.close(fig)


def fig_runtime():
    p = os.path.join(RES, "runtime.csv")
    if not os.path.exists(p):
        return
    r = pd.read_csv(p).set_index("stage")
    main = read_all("main").groupby("model")["macro_F1"].mean()
    e2e = {k: r.loc[f"END-TO-END {k}", "ms_per_event"] for k in ["CatBoost", "SBGA-F", "BiRAGNN", "SBGA-F-Stack"]}
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.7), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    for k, v in e2e.items():
        ax.scatter(v, main[k], s=45, color=COL[k], marker=MARK[k])
        ax.annotate(k, (v, main[k]), textcoords="offset points", xytext=(4, -9 if k != "SBGA-F-Stack" else 4), fontsize=7.5)
    ax.set_xlim(0.003, 0.0205)
    ax.set_ylim(96.2, 99.2)
    ax.set_xlabel("End-to-end latency (ms/event)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.grid(alpha=0.3)
    ax = axes[1]
    stages = ["Field encoding and scaling", "Flow-statistics aggregation", "Window preparation", "SBGA-F forward (GPU, batch 4096)",
              "CatBoost prediction (CPU)", "Stacking meta-learner"]
    labs = ["Encoding", "Flow stats", "Windows", "SBGA-F", "CatBoost", "Meta"]
    vals = [r.loc[s, "ms_per_event"] for s in stages]
    ax.barh(labs[::-1], vals[::-1], color="#56B4E9")
    ax.set_xlabel("ms/event")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_runtime.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_main(); fig_perturb(); fig_transfer(); fig_runtime()
    print("figures written")
