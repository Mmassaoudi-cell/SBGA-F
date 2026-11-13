"""Round-2 figures (manuscript width ~5.2 in, labels >= 7.5 pt): confusion matrix of the stack,
ablation forest plot (standalone and stack level), leakage-control summary, TON_IoT benchmark."""
import glob, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.join(os.path.dirname(__file__), "..")
R1, R2 = os.path.join(ROOT, "results", "rev"), os.path.join(ROOT, "results", "rev2")
FIG = os.path.join(ROOT, "revision", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "savefig.bbox": "tight", "pdf.fonttype": 42})
CLASSES = ["Backdoor", "DDoS_HTTP", "DDoS_ICMP", "DDoS_TCP", "DDoS_UDP", "Fingerprinting", "MITM", "Normal", "Password",
           "Port_Scanning", "Ransomware", "SQL_injection", "Uploading", "Vulnerability_scanner", "XSS"]
SHORT = ["Backdoor", "DDoS-HTTP", "DDoS-ICMP", "DDoS-TCP", "DDoS-UDP", "Fingerpr.", "MITM", "Normal", "Password",
         "Port scan", "Ransomw.", "SQL inj.", "Uploading", "Vuln. scan", "XSS"]


def read(dirn, prefix):
    fs = [f for f in sorted(glob.glob(os.path.join(dirn, f"{prefix}_*.csv"))) if os.path.basename(f)[len(prefix) + 1:-4].isdigit()]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def fig_confusion():
    p = os.path.join(R2, "cm_SBGA-F-Stack.npy")
    if not os.path.exists(p):
        return
    cm = np.load(p).astype(float)
    rn = cm / cm.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    im = ax.imshow(rn, cmap="Blues", vmin=0, vmax=100)
    for i in range(15):
        for j in range(15):
            v = rn[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.1f}" if v < 99.95 else "100", ha="center", va="center", fontsize=6.2,
                        color="white" if v > 55 else "black")
    ax.set_xticks(range(15)); ax.set_yticks(range(15))
    ax.set_xticklabels(SHORT, rotation=55, ha="right", fontsize=7.5)
    ax.set_yticklabels([f"{s} ({int(n)})" for s, n in zip(SHORT, cm.sum(1) / 10)], fontsize=7.5)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class (test events)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("% of true class", fontsize=8)
    fig.savefig(os.path.join(FIG, "fig_confusion.pdf")); plt.close(fig)


def fig_ablation():
    d = read(R2, "ablation")
    if d.empty:
        return
    main = read(R1, "main")
    ref = {"standalone": main[main.model == "SBGA-F"].set_index("seed")["macro_F1"],
           "stack": main[main.model == "SBGA-F-Stack"].set_index("seed")["macro_F1"]}
    arms = [("no_graph", "Temporal encoder only"), ("no_bidirectional", "Incoming attention only"), ("out_only", "Outgoing attention only"),
            ("no_structural", "No structural descriptors"), ("dense_aggregation", "Dense aggregation (no attention)"),
            ("k2", "$k=2$"), ("k4", "$k=4$"), ("k12", "$k=12$"), ("kall", "All neighbors (no top-$k$)"),
            ("TCN_encoder", "TCN instead of LSTM"), ("CE_loss", "Cross-entropy loss")]
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 3.5), sharey=True)
    for ax, level, title in zip(axes, ("standalone", "stack"), ("Standalone SBGA-F", "Stacked with CatBoost")):
        for i, (arm, _) in enumerate(arms):
            s = d[(d.arm == arm) & (d.level == level)].set_index("seed")["macro_F1"]
            j = s.index.intersection(ref[level].index)
            diff = (s[j] - ref[level][j]).to_numpy()
            if len(diff) < 2:
                continue
            h = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
            ax.errorbar(diff.mean(), i, xerr=h, fmt="o", ms=4, color="#0072B2" if level == "standalone" else "#B22222", capsize=2.5, lw=1.2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("$\\Delta$ macro-F1 vs. full model (points)", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(range(len(arms))); axes[0].set_yticklabels([a[1] for a in arms], fontsize=7.8)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_ablation.pdf")); plt.close(fig)


def fig_leakage():
    main = read(R1, "main"); ca = read(R2, "causal")
    ip = os.path.join(R2, "audit_interleaved_raw.csv")
    if ca.empty or not os.path.exists(ip):
        return
    it = pd.read_csv(ip)
    abl = read(R2, "ablation")
    def g(df, col, name):
        s = df[df[col] == name]["macro_F1"]
        return s.mean(), s.std(ddof=1)
    rows = {
        "CatBoost": [g(main, "model", "CatBoost"), g(it, "model", "CatBoost"), g(ca, "model", "CatBoost")],
        "Temporal-only": [g(abl[abl.level == "standalone"], "arm", "no_graph"), g(it, "model", "SBGA-F_no_graph"), g(ca, "model", "SBGA-F_no_graph")],
        "SBGA-F": [g(main, "model", "SBGA-F"), g(it, "model", "SBGA-F"), g(ca, "model", "SBGA-F")],
        "SBGA-F-Stack": [g(main, "model", "SBGA-F-Stack"), g(it, "model", "SBGA-F + CatBoost (stacked)"), g(ca, "model", "SBGA-F-Stack")],
    }
    col = {"CatBoost": "#009E73", "Temporal-only": "#999999", "SBGA-F": "#D55E00", "SBGA-F-Stack": "#B22222"}
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    w = 0.2
    for i, (m, v) in enumerate(rows.items()):
        mu, sd = [x[0] for x in v], [x[1] for x in v]
        ax.bar(np.arange(3) + (i - 1.5) * w, mu, w, yerr=sd, capsize=2, color=col[m], label=m)
        for x, y in zip(np.arange(3) + (i - 1.5) * w, mu):
            ax.text(x, 93.15, f"{y:.1f}", ha="center", va="bottom", fontsize=6.8, rotation=90, color="white")
    ax.set_xticks(range(3)); ax.set_xticklabels(["Capture order\n(primary)", "Interleaved\nstream", "Causal flow\nfeatures"], fontsize=8)
    ax.set_ylim(93, 100); ax.set_ylabel("Macro-F1 (%)")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5, columnspacing=1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(FIG, "fig_leakage.pdf")); plt.close(fig)


def fig_ton():
    d = read(R2, "ton")
    if d.empty:
        return
    order = ["CatBoost", "LightGBM", "XGBoost", "RandomForest", "RNN", "GNN", "BiRAGNN", "SBGA-F temporal-only", "SBGA-F",
             "RNN + CatBoost", "BiRAGNN + CatBoost", "Temporal-only + CatBoost", "SBGA-F-Stack"]
    order = [o for o in order if o in set(d.model)]
    col = {"CatBoost": "#009E73", "LightGBM": "#009E73", "XGBoost": "#009E73", "RandomForest": "#009E73", "RNN": "#56B4E9", "GNN": "#CC79A7",
           "BiRAGNN": "#CC79A7", "SBGA-F temporal-only": "#999999", "SBGA-F": "#D55E00"}
    agg = d.groupby("model")["macro_F1"].agg(["mean", "std"]).loc[order][::-1]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.barh(agg.index, agg["mean"], xerr=agg["std"], capsize=2.5, height=0.7, color=[col.get(m, "#B22222") for m in agg.index])
    lo = np.floor((agg["mean"] - agg["std"]).min() - 1)
    for i, (m, s) in enumerate(zip(agg["mean"], agg["std"])):
        ax.text(m + s + 0.1, i, f"{m:.2f}", va="center", fontsize=7.5)
    ax.set_xlim(lo, min(101.5, agg["mean"].max() + 2.2))
    ax.set_xlabel("Test macro-F1 (%), mean $\\pm$ s.d. over 10 seeds, TON_IoT")
    ax.tick_params(axis="y", labelsize=7.8)
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(os.path.join(FIG, "fig_ton.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig_confusion(); fig_ablation(); fig_leakage(); fig_ton()
    print("r2 figures written")
