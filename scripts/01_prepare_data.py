"""
Data preparation pipeline for the EV-GNN-IDS reproduction/new-model study.

Substitutes Edge-IIoTset (real, public, multiclass IoT/IIoT attack traffic) for the
source paper's non-reproducible EVCCS/MiniV2G dataset, per DATA_AUDIT.md.

Produces (all under project/data/):
  - packet_level.parquet   (per-packet feature rows, node_id, split, labels)
  - flow_level.parquet     (per-flow aggregated feature rows, node_id, split, labels)
  - fusion.parquet         (packet rows enriched with their flow's aggregate stats)
  - graph_{8,50,100}.json  (directed weighted adjacency: transition prob X->Y, TRAIN-only)
  - DATA_SPLIT_MANIFEST.csv

Split policy: capture/session-aware temporal split. Edge-IIoTset's 15 classes are each
stored as one contiguous chronological block (verified: 15 label-change points for 15
classes). Within each class block we split the FIRST 70% of rows -> train, NEXT 10% ->
val, LAST 20% -> test, by original row order (== capture chronology). This avoids
adjacent near-duplicate-packet leakage that a random row split would cause.

Graph node assignment: EVSC-analog "nodes" = top-K most frequent internal ip.src_host
values (K in {8,50,100}); all other hosts are folded into whichever top-K host shares
the longest IP-prefix (i.e., same /24 subnet), else assigned by hash. Standalone (K=1)
has one node / no graph, matching the source paper's standalone system.
"""
import json
import os
import numpy as np
import pandas as pd

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

SRC = r"C:\Users\MMASSAOUDI\Desktop\Data\Edge-IIoTset\Edge-IIoTset dataset\Selected dataset for ML and DL\DNN-EdgeIIoT-dataset.csv"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)

# Columns the dataset authors themselves flag as leaky / non-generalizable / high-cardinality
# (their own published preprocessing recipe, see DATA_AUDIT.md); we drop these from the
# MODEL FEATURE MATRIX but keep frame.time / ip.src_host / ip.dst_host temporarily for
# split ordering and graph construction, then drop them before training.
DROP_FOR_MODELING = [
    "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
    "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp",
    "http.request.uri.query", "tcp.options", "tcp.payload", "tcp.srcport",
    "tcp.dstport", "udp.port", "mqtt.msg", "frame.time",
]

# Per-class sampling for DL tractability, documented (not silent). A flat per-class
# cap would distort realistic imbalance (it would drag Normal, the true majority class
# with 1.6M rows, down to the same size as a mid-frequency attack). Instead we use
# sqrt-frequency compression: target_n_c ~ TOTAL_BUDGET * sqrt(n_c) / sum(sqrt(n_i)).
# This compresses the dynamic range for tractability while PRESERVING RANK ORDER
# (Normal stays the largest class, MITM/Fingerprinting stay the rarest), keeping the
# imbalance realistic -- which is the point (macro-F1 / minority recall should matter).
TOTAL_BUDGET = 300000

SYSTEM_SIZES = [1, 8, 50, 100]  # 1 == standalone (no graph)


def log(msg):
    print(f"[prepare_data] {msg}", flush=True)


def load_raw():
    log("loading raw CSV (this can take a couple minutes for 2.2M rows)...")
    df = pd.read_csv(SRC, low_memory=False)
    log(f"raw shape: {df.shape}")
    return df


def clean(df):
    before = len(df)
    df = df.drop_duplicates(keep="first")
    df = df.dropna(axis=0, how="any")
    log(f"dedup+dropna: {before} -> {len(df)} rows ({before - len(df)} removed)")
    return df


def stratified_temporal_sample(df):
    """Per-class contiguous-block temporal split, then sqrt-frequency-compressed
    sampling applied WITHIN each split (so we never sample across the temporal
    boundary, and never change the 70/10/20 ratio within a class)."""
    totals = df["Attack_type"].value_counts()
    sqrt_w = totals.map(np.sqrt)
    target_totals = (TOTAL_BUDGET * sqrt_w / sqrt_w.sum())
    class_fraction = (target_totals / totals).clip(upper=1.0)  # fraction to KEEP per class

    parts = []
    manifest_rows = []
    for atype, block in df.groupby("Attack_type", sort=False):
        n = len(block)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.10))
        frac = float(class_fraction[atype])
        block = block.copy()
        block["split"] = "test"
        block.iloc[:n_train, block.columns.get_loc("split")] = "train"
        block.iloc[n_train:n_train + n_val, block.columns.get_loc("split")] = "val"

        for split_name in ["train", "val", "test"]:
            sub = block[block["split"] == split_name]
            n_sub = len(sub)
            n_keep = max(1, int(round(n_sub * frac))) if n_sub > 0 else 0
            if n_keep < n_sub:
                sub = sub.sample(n=n_keep, random_state=RNG_SEED)
            parts.append(sub)
            manifest_rows.append({
                "Attack_type": atype, "split": split_name,
                "n_available": n_sub, "n_kept": len(sub),
                "sample_fraction": round(frac, 4),
            })
    out = pd.concat(parts, ignore_index=False).sort_index()
    manifest = pd.DataFrame(manifest_rows)
    return out, manifest


def assign_nodes(df, k):
    """Assign each row a node_id for a K-node graph system, using ip.src_host."""
    if k == 1:
        return pd.Series(0, index=df.index, name=f"node_{k}")
    train_mask = df["split"] == "train"
    top_hosts = (
        df.loc[train_mask, "ip.src_host"].value_counts().head(k).index.tolist()
    )
    top_set = set(top_hosts)
    host_to_node = {h: i for i, h in enumerate(top_hosts)}

    def fold(ip):
        if ip in top_set:
            return host_to_node[ip]
        # fold into nearest top-K host by longest shared dotted-prefix (same /24, /16, /8)
        best, best_score = 0, -1
        ip_parts = str(ip).split(".")
        for h in top_hosts:
            h_parts = h.split(".")
            score = 0
            for a, b in zip(ip_parts, h_parts):
                if a == b:
                    score += 1
                else:
                    break
            if score > best_score:
                best_score = score
                best = host_to_node[h]
        return best

    return df["ip.src_host"].map(fold).rename(f"node_{k}")


def build_graph(df, k, node_col):
    """TRAIN-only empirical transition probability graph: P(next row's node = Y |
    current row's node = X), consecutive in temporal (row) order, mirroring the
    source paper's probabilistic-charging-graph estimation from mobility logs."""
    train = df[df["split"] == "train"].sort_index()
    nodes = train[node_col].to_numpy()
    counts = np.zeros((k, k), dtype=np.float64)
    for x, y in zip(nodes[:-1], nodes[1:]):
        counts[x, y] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = counts / row_sums
    edges = []
    for i in range(k):
        for j in range(k):
            if probs[i, j] > 0:
                edges.append({"src": int(i), "dst": int(j), "weight": float(probs[i, j])})
    return {"n_nodes": k, "edges": edges}


MAX_FLOW_PKTS = 25  # NetFlow-style active-timeout analog: cap packets per flow record


def build_flow_level(df):
    """Aggregate packets sharing (split, Attack_type, src_host, dst_host) into flow
    records, chunked to at most MAX_FLOW_PKTS consecutive (temporally ordered) packets
    per record -- a NetFlow-style "active timeout" analog. Without this cap, classes
    dominated by one or two persistent host pairs (verified: Normal traffic collapses
    to <20 total flow records across all three splits) produce a near-degenerate
    flow-level table with too few benign examples to evaluate (confirmed empirically:
    FA=100% / macro-F1~50% on an initial run -- majority-class collapse). Chunking
    fixes this while still mirroring the source paper's flow-level table being much
    smaller/coarser than its packet-level table. Flow keys are computed independently
    per split, so no flow record can straddle train/val/test."""
    d = df.sort_index().copy()
    base_key = (
        d["split"].astype(str) + "|" + d["Attack_type"].astype(str) + "|" +
        d["ip.src_host"].astype(str) + "|" + d["ip.dst_host"].astype(str)
    )
    rank_in_key = d.groupby(base_key).cumcount()
    chunk_key = base_key + "|" + (rank_in_key // MAX_FLOW_PKTS).astype(str)
    d["_flow_key_change"] = chunk_key.astype("category").cat.codes

    def parse_time(s):
        try:
            t = str(s).strip().split(" ")[-1]
            h, m, sec = t.split(":")
            return int(h) * 3600 + int(m) * 60 + float(sec)
        except Exception:
            return np.nan

    d["_t"] = d["frame.time"].map(parse_time)
    d["_pktlen"] = d[["tcp.len", "http.content_length", "mqtt.len"]].fillna(0).max(axis=1)

    agg = d.groupby("_flow_key_change").agg(
        node_8=("node_8", "first"),
        node_50=("node_50", "first"),
        node_100=("node_100", "first"),
        node_1=("node_1", "first"),
        split=("split", "first"),
        Attack_label=("Attack_label", "first"),
        Attack_type=("Attack_type", "first"),
        start_time=("_t", "min"),
        end_time=("_t", "max"),
        total_packets=("_t", "count"),
        total_bytes=("_pktlen", "sum"),
        max_packet_length=("_pktlen", "max"),
    )
    agg["flow_duration"] = (agg["end_time"] - agg["start_time"]).clip(lower=0).fillna(0)
    iats = d.groupby("_flow_key_change")["_t"].apply(lambda s: np.diff(np.sort(s.values)))
    agg["mean_flow_iat"] = iats.map(lambda a: float(np.mean(a)) if len(a) else 0.0)
    agg["min_flow_iat"] = iats.map(lambda a: float(np.min(a)) if len(a) else 0.0)
    agg["max_flow_iat"] = iats.map(lambda a: float(np.max(a)) if len(a) else 0.0)
    agg["pps"] = agg["total_packets"] / agg["flow_duration"].replace(0, np.nan)
    agg["pps"] = agg["pps"].fillna(agg["total_packets"])
    agg = agg.reset_index()  # keep _flow_key_change as a column (join key for fusion)
    return agg


def main():
    df = load_raw()
    df = clean(df)
    df, manifest = stratified_temporal_sample(df)
    log(f"post temporal-split+cap sampling shape: {df.shape}")

    for k in SYSTEM_SIZES:
        df[f"node_{k}"] = assign_nodes(df, k)

    # ---- packet-level table ----
    packet_cols = [c for c in df.columns if c not in DROP_FOR_MODELING]
    packet_df = df[packet_cols].copy()
    packet_path = os.path.join(OUT_DIR, "packet_level.parquet")
    packet_df.to_parquet(packet_path, index=False)
    log(f"saved packet-level table: {packet_df.shape} -> {packet_path}")

    # ---- flow-level table ----
    flow_df = build_flow_level(df)
    flow_path = os.path.join(OUT_DIR, "flow_level.parquet")
    flow_df.to_parquet(flow_path, index=False)
    log(f"saved flow-level table: {flow_df.shape} -> {flow_path}")

    # ---- fusion table: packet rows + their flow's aggregate stats ----
    d2 = df.sort_index().copy()
    base_key2 = (
        d2["split"].astype(str) + "|" + d2["Attack_type"].astype(str) + "|" +
        d2["ip.src_host"].astype(str) + "|" + d2["ip.dst_host"].astype(str)
    )
    rank2 = d2.groupby(base_key2).cumcount()
    chunk_key2 = base_key2 + "|" + (rank2 // MAX_FLOW_PKTS).astype(str)
    d2["_flow_key_change"] = chunk_key2.astype("category").cat.codes
    flow_stats = flow_df  # already has _flow_key_change as a column (join key)
    keep_flow_cols = ["_flow_key_change", "flow_duration", "mean_flow_iat", "min_flow_iat",
                       "max_flow_iat", "pps", "total_packets", "total_bytes", "max_packet_length"]
    fusion_df = d2.merge(flow_stats[keep_flow_cols], on="_flow_key_change", how="left")
    fusion_df = fusion_df[[c for c in fusion_df.columns if c not in DROP_FOR_MODELING and c != "_flow_key_change"]]
    fusion_path = os.path.join(OUT_DIR, "fusion.parquet")
    fusion_df.to_parquet(fusion_path, index=False)
    log(f"saved fusion table: {fusion_df.shape} -> {fusion_path}")

    # ---- graphs ----
    for k in [8, 50, 100]:
        g = build_graph(df, k, f"node_{k}")
        with open(os.path.join(OUT_DIR, f"graph_{k}.json"), "w") as f:
            json.dump(g, f)
        log(f"saved graph_{k}.json: {len(g['edges'])} directed edges")

    # ---- manifest ----
    manifest_path = os.path.join(os.path.dirname(__file__), "..", "DATA_SPLIT_MANIFEST.csv")
    manifest.to_csv(manifest_path, index=False)
    log(f"saved split manifest -> {manifest_path}")

    # near-duplicate audit (rounded-numeric-feature Jaccard collision rate, sampled)
    numeric_cols = packet_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("Attack_label",) and not c.startswith("node_")]
    tr = packet_df[packet_df["split"] == "train"][numeric_cols].round(2)
    te = packet_df[packet_df["split"] == "test"][numeric_cols].round(2)
    tr_hashes = set(map(tuple, tr.sample(min(20000, len(tr)), random_state=1).to_numpy()))
    te_sample = te.sample(min(20000, len(te)), random_state=1)
    te_hashes = list(map(tuple, te_sample.to_numpy()))
    collisions = sum(1 for h in te_hashes if h in tr_hashes)
    log(f"near-dup audit: {collisions}/{len(te_hashes)} rounded-feature collisions between sampled train/test rows")

    with open(os.path.join(os.path.dirname(__file__), "..", "data_audit", "near_dup_audit.txt"), "w") as f:
        f.write(f"rounded-feature exact-collision rate on sampled rows: {collisions}/{len(te_hashes)} "
                f"= {collisions/len(te_hashes):.4%}\n")

    log("done.")


if __name__ == "__main__":
    main()
