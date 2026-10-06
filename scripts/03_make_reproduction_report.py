"""Builds REPRODUCTION_REPORT.md from results/reproduction_results.csv, comparing
against the source paper's own published Table III/IV numbers (hardcoded from
SOURCE_PAPER_AUDIT.md -- these are literal transcriptions of the published tables,
never altered)."""
import os
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "REPRODUCTION_REPORT.md")

# Published numbers, Table III (standalone) and Table IV (N-node), AC/DR/FA in %.
PUBLISHED = {
    # (system, view, model): (AC, DR, FA)
    (1, "packet_level", "FNN"): (92.5, 93.3, 10.1), (1, "flow_level", "FNN"): (91.5, 92.7, 11.6), (1, "fusion", "FNN"): (92.8, 93.6, 9.8),
    (1, "packet_level", "RNN"): (93.2, 94.5, 8.0), (1, "flow_level", "RNN"): (92.2, 93.4, 9.5), (1, "fusion", "RNN"): (93.6, 94.9, 7.6),
    (1, "packet_level", "CNN"): (94.0, 95.1, 7.2), (1, "flow_level", "CNN"): (93.4, 94.7, 8.0), (1, "fusion", "CNN"): (94.3, 95.4, 6.7),
    (1, "packet_level", "Transformer"): (94.2, 95.6, 6.9), (1, "flow_level", "Transformer"): (94.1, 95.0, 7.8), (1, "fusion", "Transformer"): (94.7, 95.9, 6.5),

    (8, "packet_level", "FNN"): (89.4, 90.5, 17.8), (8, "flow_level", "FNN"): (89.1, 89.9, 18.3), (8, "fusion", "FNN"): (90.3, 91.2, 16.8),
    (8, "packet_level", "RNN"): (90.3, 91.4, 14.4), (8, "flow_level", "RNN"): (90.5, 91.8, 13.5), (8, "fusion", "RNN"): (90.9, 91.9, 12.7),
    (8, "packet_level", "CNN"): (91.9, 93.2, 10.9), (8, "flow_level", "CNN"): (91.3, 92.1, 11.8), (8, "fusion", "CNN"): (92.4, 93.3, 10.2),
    (8, "packet_level", "Transformer"): (92.0, 93.4, 11.1), (8, "flow_level", "Transformer"): (91.5, 92.9, 11.5), (8, "fusion", "Transformer"): (92.3, 93.5, 10.8),
    (8, "packet_level", "GNN"): (92.3, 93.5, 10.5), (8, "flow_level", "GNN"): (91.6, 92.3, 11.4), (8, "fusion", "GNN"): (92.5, 93.7, 10.0),
    (8, "packet_level", "TGNN"): (92.7, 93.8, 10.1), (8, "flow_level", "TGNN"): (91.8, 92.4, 11.2), (8, "fusion", "TGNN"): (92.5, 94.0, 9.8),
    (8, "packet_level", "BiRAGNN"): (95.4, 95.8, 5.5), (8, "flow_level", "BiRAGNN"): (94.3, 94.7, 5.9), (8, "fusion", "BiRAGNN"): (96.0, 96.1, 4.9),

    (50, "packet_level", "FNN"): (89.2, 90.1, 14.7), (50, "flow_level", "FNN"): (89.4, 90.7, 13.7), (50, "fusion", "FNN"): (90.5, 91.3, 13.2),
    (50, "packet_level", "RNN"): (91.0, 92.2, 9.9), (50, "flow_level", "RNN"): (91.2, 92.5, 9.5), (50, "fusion", "RNN"): (91.3, 93.1, 9.3),
    (50, "packet_level", "CNN"): (92.1, 94.4, 8.2), (50, "flow_level", "CNN"): (92.2, 93.5, 8.0), (50, "fusion", "CNN"): (92.6, 94.5, 7.8),
    (50, "packet_level", "Transformer"): (92.3, 94.6, 8.1), (50, "flow_level", "Transformer"): (92.5, 92.9, 7.9), (50, "fusion", "Transformer"): (92.9, 94.6, 7.5),
    (50, "packet_level", "GNN"): (93.6, 94.7, 7.9), (50, "flow_level", "GNN"): (92.5, 93.9, 7.5), (50, "fusion", "GNN"): (93.4, 94.9, 7.0),
    (50, "packet_level", "TGNN"): (93.8, 94.9, 7.7), (50, "flow_level", "TGNN"): (92.7, 94.2, 7.3), (50, "fusion", "TGNN"): (93.4, 95.1, 7.0),
    (50, "packet_level", "BiRAGNN"): (96.5, 96.8, 4.3), (50, "flow_level", "BiRAGNN"): (94.9, 95.1, 3.9), (50, "fusion", "BiRAGNN"): (97.0, 97.1, 3.4),

    (100, "packet_level", "FNN"): (90.2, 91.2, 12.3), (100, "flow_level", "FNN"): (89.1, 90.3, 12.8), (100, "fusion", "FNN"): (91.0, 91.5, 12.0),
    (100, "packet_level", "RNN"): (92.2, 93.4, 10.9), (100, "flow_level", "RNN"): (93.0, 93.9, 10.1), (100, "fusion", "RNN"): (93.8, 94.2, 9.8),
    (100, "packet_level", "CNN"): (92.8, 94.1, 7.6), (100, "flow_level", "CNN"): (93.1, 93.9, 7.3), (100, "fusion", "CNN"): (93.9, 95.0, 7.1),
    (100, "packet_level", "Transformer"): (93.2, 94.6, 7.6), (100, "flow_level", "Transformer"): (92.9, 94.1, 7.2), (100, "fusion", "Transformer"): (93.9, 95.1, 7.0),
    (100, "packet_level", "GNN"): (93.2, 94.6, 7.3), (100, "flow_level", "GNN"): (92.9, 94.3, 7.0), (100, "fusion", "GNN"): (94.0, 95.3, 6.9),
    (100, "packet_level", "TGNN"): (94.1, 95.2, 7.1), (100, "flow_level", "TGNN"): (93.3, 94.3, 6.8), (100, "fusion", "TGNN"): (94.2, 95.6, 6.5),
    (100, "packet_level", "BiRAGNN"): (98.3, 98.5, 3.3), (100, "flow_level", "BiRAGNN"): (97.4, 97.6, 3.3), (100, "fusion", "BiRAGNN"): (98.7, 98.8, 2.9),
}


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "reproduction_results.csv"))
    lines = []
    lines.append("# REPRODUCTION_REPORT.md\n")
    lines.append("**Published** = source paper Tables III/IV (transcribed verbatim). "
                  "**Reproduced** = this project's local run of the paper's architecture/protocol on "
                  "*substituted data* (Edge-IIoTset, see DATA_AUDIT.md) — never the same underlying dataset, "
                  "so absolute-value agreement is not expected or claimed. What we assess is whether the "
                  "*qualitative pattern* the paper reports (BiRAGNN > TGNN/GNN > Transformer/CNN > RNN/FNN; "
                  "fusion > flow > packet; larger graphs help BiRAGNN more than benchmarks) survives on "
                  "independent, real data.\n")

    for system in [1, 8, 50, 100]:
        sys_label = "Standalone" if system == 1 else f"{system}-node"
        lines.append(f"\n## {sys_label} system\n")
        lines.append("| Model | View | Published AC | Reproduced AC | ΔAC | Published DR | Reproduced DR | ΔDR | Published FA | Reproduced FA | ΔFA |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        sub = df[df.system_size == system]
        for _, row in sub.iterrows():
            key = (system, row["view"], row["model"])
            if key not in PUBLISHED:
                continue
            pac, pdr, pfa = PUBLISHED[key]
            dac, ddr, dfa = row["AC"] - pac, row["DR"] - pdr, row["FA"] - pfa
            lines.append(f"| {row['model']} | {row['view']} | {pac:.1f} | {row['AC']:.1f} | {dac:+.1f} | "
                         f"{pdr:.1f} | {row['DR']:.1f} | {ddr:+.1f} | {pfa:.1f} | {row['FA']:.1f} | {dfa:+.1f} |")

    # extra metrics table (not in the paper at all: macro-F1, PR-AUC, params, timing)
    lines.append("\n## Additional metrics not reported in the source paper (macro-F1, PR-AUC, efficiency)\n")
    lines.append("| System | View | Model | macro-F1 | weighted-F1 | PR-AUC | Params | Train (s) | Infer (ms/sample) |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for _, row in df.sort_values(["system_size", "view", "model"]).iterrows():
        lines.append(f"| {row['system_size']} | {row['view']} | {row['model']} | {row['macro_F1']:.1f} | "
                     f"{row['weighted_F1']:.1f} | {row['PR_AUC']:.3f} | {row['n_params']:,} | "
                     f"{row['train_time_sec']:.1f} | {row['infer_time_ms_per_sample']:.4f} |")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
