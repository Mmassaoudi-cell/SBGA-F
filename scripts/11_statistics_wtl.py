"""
Statistical validation (Section 22) and win/tie/loss analysis (Section 30) of the
final model vs. every benchmark, on final_test_results.csv (5 seeds each).

For each benchmark: paired Wilcoxon signed-rank test on per-seed macro-F1 (proposed
vs. benchmark), Holm-corrected across the benchmark family; classification into
statistically superior / practically superior (>1 pt mean diff, not significant) /
statistically indistinguishable / inferior, plus effect size (rank-biserial correlation).

Outputs: STATISTICAL_TESTS.csv, BENCHMARK_WTL.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")


def rank_biserial(diffs):
    """Effect size for Wilcoxon signed-rank: matched-pairs rank-biserial correlation."""
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diffs))
    pos = ranks[diffs > 0].sum()
    neg = ranks[diffs < 0].sum()
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "final_test_results.csv"))
    proposed = df[df.model == "Proposed_SBGA-F"].set_index("seed")["macro_F1"]
    benchmarks = [m for m in df["model"].unique() if m != "Proposed_SBGA-F"]

    rows = []
    for bm in benchmarks:
        bm_series = df[df.model == bm].set_index("seed")["macro_F1"]
        common_seeds = sorted(set(proposed.index) & set(bm_series.index))
        if len(common_seeds) < 3:
            continue
        p_vals = proposed.loc[common_seeds].to_numpy()
        b_vals = bm_series.loc[common_seeds].to_numpy()
        diffs = p_vals - b_vals
        mean_diff = diffs.mean()
        try:
            stat, pval = stats.wilcoxon(p_vals, b_vals, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            pval = 1.0  # all-zero differences
        eff = rank_biserial(diffs)
        cat = df[df.model == bm]["category"].iloc[0]
        rows.append(dict(benchmark=bm, category=cat, proposed_mean=p_vals.mean(), benchmark_mean=b_vals.mean(),
                          mean_diff=mean_diff, wilcoxon_p=pval, effect_size_rank_biserial=eff, n_seeds=len(common_seeds)))

    stats_df = pd.DataFrame(rows)
    # Holm correction across the benchmark family
    reject, p_holm, _, _ = multipletests(stats_df["wilcoxon_p"], alpha=0.05, method="holm")
    stats_df["p_holm"] = p_holm
    stats_df["significant_holm"] = reject

    def classify(row):
        if row["significant_holm"] and row["mean_diff"] > 0:
            return "statistically_superior"
        if row["significant_holm"] and row["mean_diff"] < 0:
            return "statistically_inferior"
        if not row["significant_holm"] and abs(row["mean_diff"]) >= 1.0:
            return "practically_superior" if row["mean_diff"] > 0 else "practically_inferior"
        return "statistically_indistinguishable"

    stats_df["verdict"] = stats_df.apply(classify, axis=1)
    stats_df = stats_df.sort_values("mean_diff", ascending=False)
    stats_df.to_csv(os.path.join(RESULTS_DIR, "STATISTICAL_TESTS.csv"), index=False)

    # Win/tie/loss summary
    wins = (stats_df["verdict"].isin(["statistically_superior", "practically_superior"])).sum()
    losses = (stats_df["verdict"].isin(["statistically_inferior", "practically_inferior"])).sum()
    ties = (stats_df["verdict"] == "statistically_indistinguishable").sum()

    wtl_rows = stats_df[["benchmark", "category", "proposed_mean", "benchmark_mean", "mean_diff",
                          "wilcoxon_p", "p_holm", "effect_size_rank_biserial", "verdict"]].copy()
    wtl_rows.to_csv(os.path.join(PROJECT_DIR, "BENCHMARK_WTL.csv"), index=False)

    print(stats_df.to_string(index=False))
    print(f"\nProposed_SBGA-F: {wins} wins / {ties} ties / {losses} losses (out of {len(stats_df)} benchmarks)")

    with open(os.path.join(PROJECT_DIR, "results", "wtl_summary.txt"), "w") as f:
        f.write(f"Proposed_SBGA-F: {wins} wins / {ties} ties / {losses} losses (out of {len(stats_df)} benchmarks)\n")


if __name__ == "__main__":
    main()
