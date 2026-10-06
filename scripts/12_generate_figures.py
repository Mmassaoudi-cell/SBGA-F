"""Generates the manuscript's figures 3-6 directly from the saved result CSVs
(Section 31/34: figures/tables must come from real result files, never typed by hand)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.bbox": "tight"})
COLORS = {"proposed": "#D55E00", "source_method": "#0072B2", "boosting": "#009E73",
          "classical": "#999999", "deep": "#56B4E9", "graph": "#CC79A7"}


def fig3_main_benchmark():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "final_test_results.csv"))
    agg = df.groupby(["model", "category"])["macro_F1"].agg(["mean", "std"]).reset_index()
    agg = agg.sort_values("mean", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [COLORS.get(c, "#333333") for c in agg["category"]]
    ax.barh(agg["model"], agg["mean"], xerr=agg["std"], color=colors, capsize=3)
    ax.set_xlabel("Macro-F1 on held-out TEST set (%, mean ± std over 5 seeds)")
    ax.set_xlim(80, 100)
    ax.axvline(agg.loc[agg.model == "Proposed_SBGA-F", "mean"].values[0], color=COLORS["proposed"],
               linestyle="--", linewidth=1, alpha=0.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in COLORS.values()]
    ax.legend(handles, COLORS.keys(), loc="lower right", fontsize=8, title="category")
    ax.set_title("15-class malware/attack detection: proposed model vs. 15 benchmarks")
    fig.savefig(os.path.join(FIG_DIR, "fig3_main_benchmark.pdf"))
    plt.close(fig)


def fig4_cross_topology():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "robustness_generalization_results.csv"))
    ct = df[df.experiment == "cross_topology_transfer"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sizes = [8, 50]
    x = np.arange(len(sizes))
    width = 0.35
    zeroshot = [ct[(ct.target_system == s) & (ct["mode"] == "zero_shot_from_100node")]["macro_F1"].values[0] for s in sizes]
    retrained = [ct[(ct.target_system == s) & (ct["mode"] == "retrained_on_target")]["macro_F1"].values[0] for s in sizes]
    ax.bar(x - width / 2, zeroshot, width, label="Zero-shot transfer from 100-node model", color=COLORS["proposed"])
    ax.bar(x + width / 2, retrained, width, label="Retrained from scratch on target graph", color="#333333")
    ax.set_xticks(x); ax.set_xticklabels([f"{s}-node graph" for s in sizes])
    ax.set_ylabel("Macro-F1 on target graph's TEST set (%)")
    ax.set_ylim(85, 100)
    ax.set_title("Cross-topology zero-shot transfer (no retraining) vs. retrained upper bound")
    ax.legend(fontsize=8, loc="lower center")
    for i, (z, r) in enumerate(zip(zeroshot, retrained)):
        ax.text(i - width / 2, z + 0.3, f"{z:.1f}", ha="center", fontsize=8)
        ax.text(i + width / 2, r + 0.3, f"{r:.1f}", ha="center", fontsize=8)
    fig.savefig(os.path.join(FIG_DIR, "fig4_cross_topology_transfer.pdf"))
    plt.close(fig)


def fig5_ablation_robustness():
    abl = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"))
    rob = pd.read_csv(os.path.join(RESULTS_DIR, "robustness_generalization_results.csv"))
    rob_bm = pd.read_csv(os.path.join(RESULTS_DIR, "robustness_baseline_comparison.csv")) if os.path.exists(
        os.path.join(RESULTS_DIR, "robustness_baseline_comparison.csv")) else None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # left: ablation bar chart
    aagg = abl.groupby("arm")["macro_F1"].agg(["mean", "std"]).reset_index().sort_values("mean")
    axes[0].barh(aagg["arm"], aagg["mean"], xerr=aagg["std"], color="#56B4E9", capsize=3)
    axes[0].set_xlabel("Macro-F1 on TEST (%, mean ± std over 3 seeds)")
    axes[0].set_xlim(93, 99)
    axes[0].set_title("(a) Ablation: SBGA-F components")

    # right: robustness curves (proposed vs BiRAGNN vs XGBoost) under feature noise
    noise_prop = rob[rob.experiment == "feature_noise"].sort_values("noise_std")
    axes[1].plot(noise_prop["noise_std"], noise_prop["macro_F1"], "o-", color=COLORS["proposed"], label="Proposed SBGA-F")
    if rob_bm is not None:
        for name, color in [("BiRAGNN", COLORS["source_method"]), ("XGBoost", COLORS["boosting"])]:
            sub = rob_bm[(rob_bm.model == name) & (rob_bm.experiment == "feature_noise")].sort_values("noise_std")
            axes[1].plot(sub["noise_std"], sub["macro_F1"], "o-", color=color, label=name)
    axes[1].set_xlabel("Injected Gaussian feature noise (std, standardized units)")
    axes[1].set_ylabel("Macro-F1 on TEST (%)")
    axes[1].set_title("(b) Robustness to feature noise")
    axes[1].legend(fontsize=8)

    fig.savefig(os.path.join(FIG_DIR, "fig5_ablation_robustness.pdf"))
    plt.close(fig)


def fig6_pareto():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "final_test_results.csv"))
    agg = df.groupby(["model", "category"]).agg(macro_F1=("macro_F1", "mean"),
                                                  infer_ms=("infer_time_ms_per_sample", "mean"),
                                                  params=("n_params", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for cat, color in COLORS.items():
        sub = agg[agg.category == cat]
        axes[0].scatter(sub["infer_ms"], sub["macro_F1"], color=color, label=cat, s=50)
        axes[1].scatter(sub["params"], sub["macro_F1"], color=color, label=cat, s=50)
    for ax_i, xcol, xlabel in [(0, "infer_ms", "Inference latency (ms/sample)"), (1, "params", "Parameters (log scale)")]:
        for _, row in agg.iterrows():
            axes[ax_i].annotate(row["model"], (row[xcol], row["macro_F1"]), fontsize=6,
                                 xytext=(3, 3), textcoords="offset points")
        axes[ax_i].set_xscale("log")
        axes[ax_i].set_xlabel(xlabel)
    axes[0].set_ylabel("Macro-F1 on TEST (%)")
    axes[0].set_title("(a) Performance vs. inference latency")
    axes[1].set_title("(b) Performance vs. model size")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.savefig(os.path.join(FIG_DIR, "fig6_pareto.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig3_main_benchmark()
    fig4_cross_topology()
    fig5_ablation_robustness()
    fig6_pareto()
    print("Saved figures to", FIG_DIR)
