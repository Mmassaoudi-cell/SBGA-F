"""Shared data-loading, windowing, and graph utilities for the reproduction study."""
import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

LABEL_COLS = {"Attack_label", "Attack_type", "split"}
NODE_COLS = {"node_1", "node_8", "node_50", "node_100"}


def load_view(view):
    path = os.path.join(DATA_DIR, f"{view}.parquet")
    df = pd.read_parquet(path)
    return df.reset_index(drop=True)  # global positions == row index, needed for windowing


def get_feature_columns(df):
    exclude = LABEL_COLS | NODE_COLS | {"_flow_key_change"}
    cols = [c for c in df.columns if c not in exclude]
    return cols


def fit_transform_features(df, feature_cols):
    """Fit encoders/scalers on TRAIN rows only, transform all rows. Returns a new
    numeric-only DataFrame (same index) plus the fitted objects."""
    df = df.copy()
    num_cols, cat_cols = [], []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    train_mask = df["split"] == "train"

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(df.loc[train_mask, cat_cols].astype(str))
        df[cat_cols] = enc.transform(df[cat_cols].astype(str))
    for c in num_cols:
        df[c] = df[c].astype(float)
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        med = df.loc[train_mask, c].median()
        df[c] = df[c].fillna(med)

    scaler = StandardScaler()
    all_feat_cols = num_cols + cat_cols
    scaler.fit(df.loc[train_mask, all_feat_cols].to_numpy())
    X = scaler.transform(df[all_feat_cols].to_numpy()).astype(np.float32)

    return X, all_feat_cols


def build_windows(df, node_col, split, W=10):
    """For rows in `split`, build a causal trailing window (length W, left-padded by
    repeating the first available row) of GLOBAL row positions (indices into the full,
    view-level feature array `df`/`X_all`), grouped per node_col, preserving original
    (temporal) row order. Windows never cross the split boundary (built independently
    per split), so no leakage. Returns: (split_positions [n], window_idx [n, W])."""
    sub = df[df["split"] == split]
    global_pos = sub.index.to_numpy()  # df has a plain RangeIndex (see load_view)
    node_vals = sub[node_col].to_numpy()

    pos_by_node = {}
    for pos, nv in zip(global_pos, node_vals):
        pos_by_node.setdefault(nv, []).append(pos)

    result = np.zeros((len(sub), W), dtype=np.int64)
    pos_to_row = {pos: i for i, pos in enumerate(global_pos)}
    for nv, positions in pos_by_node.items():
        hist = []
        for pos in positions:
            hist.append(pos)
            window = hist[-W:]
            if len(window) < W:
                window = [window[0]] * (W - len(window)) + window
            result[pos_to_row[pos]] = window
    return global_pos, result


def load_graph(k):
    path = os.path.join(DATA_DIR, f"graph_{k}.json")
    with open(path) as f:
        g = json.load(f)
    n = g["n_nodes"]
    A_in = np.zeros((n, n), dtype=np.float32)   # A_in[v, u] = weight of edge u->v (in-neighbors of v)
    for e in g["edges"]:
        A_in[e["dst"], e["src"]] = e["weight"]
    A_out = A_in.T.copy()  # A_out[v, u] = weight of edge v->u (out-neighbors of v)
    return A_in, A_out


def structural_node_features(A_in):
    """Topology-only per-node features (in-/out-degree, PageRank, clustering) computed
    from the TRAIN-only probabilistic graph. Identity-free: a model trained on these
    (instead of learned per-node-ID embeddings) can in principle be evaluated on a
    differently-sized/differently-topologied graph it never saw (used by Candidate B)."""
    import networkx as nx
    n = A_in.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for v in range(n):
        for u in range(n):
            if A_in[v, u] > 0:
                G.add_edge(u, v, weight=float(A_in[v, u]))  # u -> v
    indeg = np.array([G.in_degree(v, weight="weight") for v in range(n)], dtype=np.float32)
    outdeg = np.array([G.out_degree(v, weight="weight") for v in range(n)], dtype=np.float32)
    try:
        pr = nx.pagerank(G, weight="weight")
        pagerank = np.array([pr.get(v, 0.0) for v in range(n)], dtype=np.float32)
    except Exception:
        pagerank = np.zeros(n, dtype=np.float32)
    try:
        clus = nx.clustering(G.to_undirected(), weight="weight")
        clustering = np.array([clus.get(v, 0.0) for v in range(n)], dtype=np.float32)
    except Exception:
        clustering = np.zeros(n, dtype=np.float32)
    feats = np.stack([indeg, outdeg, pagerank, clustering], axis=1)
    # normalize (structural features have very different scales)
    feats = (feats - feats.mean(axis=0, keepdims=True)) / (feats.std(axis=0, keepdims=True) + 1e-6)
    return feats.astype(np.float32)  # [n_nodes, 4]


def topk_neighbors(A, k):
    """Per-node top-k neighbor indices and weights from a weighted adjacency A[v,u]."""
    n = A.shape[0]
    idx = np.argsort(-A, axis=1)[:, :k]
    w = np.take_along_axis(A, idx, axis=1)
    return idx.astype(np.int64), w.astype(np.float32)


def node_prototypes(X, node_ids, n_nodes):
    """Mean feature vector per node, computed from the given (TRAIN) rows only."""
    d = X.shape[1]
    proto = np.zeros((n_nodes, d), dtype=np.float32)
    counts = np.zeros(n_nodes, dtype=np.int64)
    for i, nv in enumerate(node_ids):
        proto[nv] += X[i]
        counts[nv] += 1
    counts[counts == 0] = 1
    proto /= counts[:, None]
    return proto
