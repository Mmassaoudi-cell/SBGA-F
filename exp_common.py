"""
SBGA-F model code: data interface, graph context, SBGA-F network, training, and stacking.

Implementation notes:
  * GPU-resident data: windows are stored as index arrays into the feature matrix and are
    gathered per mini-batch, so every training run is considerably faster.
  * Explicit graph edge cases:
      - top-k slots whose edge weight is zero (node has fewer than k observed neighbours)
        are masked out of the attention softmax; a node without any neighbour receives a
        zero spatial context vector,
      - a node with fewer than N_MIN_PROTO training events uses the global training mean
        as its prototype.
  * Pluggable graph construction (transition / block-shuffled / shuffled / observed
    communication link / degree-preserving rewiring) and event-stream protocol
    (capture order vs. interleaved stream).
  * Perturbation suite that respects feature types (numeric noise, categorical
    corruption, missing-value imputation, timing jitter, flow-exporter dropout, topology
    perturbations).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import load_view, get_feature_columns, structural_node_features, topk_neighbors
from models import LSTMAttnEncoder, act_layer, IDSModel, MODEL_SPECS
from candidate_models import FocalLoss, class_weights_sqrt, TCNEncoder, DenseNeighborAgg
from ft_transformer import FTTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.environ.get("SBGAF_RESULTS", os.path.join(ROOT, "results"))
SMOKE = bool(os.environ.get("SBGAF_SMOKE"))
# On this PyTorch/cuDNN build the float32 cuDNN LSTM kernel is ~2x slower than the native one.
# SBGAF_NATIVE_LSTM=1 selects the native kernel (same model, same precision).
if os.environ.get("SBGAF_NATIVE_LSTM") == "1":
    torch.backends.cudnn.enabled = False
PROB_DIR = os.path.join(RESULTS_DIR, "probs")

SEEDS10 = [42, 123, 2024, 7, 99, 555, 777, 2025, 31, 88]
W = 10
K = 6
N_MIN_PROTO = 5
HP = dict(hidden=192, layers=2, dropout=0.3397475692650171, lr=0.0018098827092128786,
          gamma=1.9229587245058637, batch=1024, max_epochs=30, patience=8)

FLOW_COLS = ["flow_duration", "mean_flow_iat", "min_flow_iat", "max_flow_iat", "pps",
             "total_packets", "total_bytes", "max_packet_length"]
TIMING_SUBSTR = ["iat", "duration", "pps", "total_bytes", "total_packets", "max_packet_length",
                 "tcp.len", "time_delta", "content_length", "mqtt.len"]


def to_t(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype)


# ----------------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------------
class Base:
    """Feature matrix (train-fitted preprocessing) and labels shared by all experiments."""
    pass


_BASE = None
_BASES = {}


def causal_flow_features(df, aux, max_pkts=25):
    """Flow aggregates that use only the packets observed up to and including the current one.
    Flows are keyed by (partition, source host, destination host) -- the class label is NOT part
    of the key -- and chunked to at most max_pkts packets in capture order."""
    t = aux["frame.time"].astype(str).str.strip().str.split(" ").str[-1].str.split(":", expand=True)
    for c in (0, 1, 2):
        t[c] = pd.to_numeric(t[c], errors="coerce") if c in t.columns else np.nan
    tt = (t[0] * 3600 + t[1] * 60 + t[2])
    key = df["split"].astype(str) + "|" + aux["ip.src_host"].astype(str) + "|" + aux["ip.dst_host"].astype(str)
    rank = key.groupby(key).cumcount()
    g = (key + "|" + (rank // max_pkts).astype(str)).astype("category").cat.codes
    tt = tt.groupby(g).ffill().fillna(0.0)
    plen = df[["tcp.len", "http.content_length", "mqtt.len"]].fillna(0).max(axis=1).astype(float)
    out = pd.DataFrame(index=df.index)
    n = tt.groupby(g).cumcount() + 1
    out["total_packets"] = n.astype(float)
    out["flow_duration"] = (tt.groupby(g).cummax() - tt.groupby(g).cummin()).clip(lower=0)
    iat = tt.groupby(g).diff().abs()
    cnt = iat.notna().groupby(g).cumsum()
    out["mean_flow_iat"] = (iat.fillna(0).groupby(g).cumsum() / cnt.replace(0, np.nan)).fillna(0.0)
    out["min_flow_iat"] = iat.groupby(g).cummin().fillna(0.0)
    out["max_flow_iat"] = iat.groupby(g).cummax().fillna(0.0)
    pps = out["total_packets"] / out["flow_duration"].replace(0, np.nan)
    out["pps"] = pps.fillna(out["total_packets"])
    out["total_bytes"] = plen.groupby(g).cumsum()
    out["max_packet_length"] = plen.groupby(g).cummax()
    return out


def load_base(variant="default"):
    global _BASE
    if variant in _BASES:
        return _BASES[variant]
    df = load_view("fusion")
    if variant == "causal":
        cf = causal_flow_features(df, load_aux())
        for c in cf.columns:
            assert c in df.columns, c
            df[c] = cf[c].to_numpy()
    cols = get_feature_columns(df)
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if c not in num_cols]
    train_mask = (df["split"] == "train").to_numpy()
    d = df[cols].copy()
    unknown_code = -1
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=unknown_code)
        enc.fit(d.loc[train_mask, cat_cols].astype(str))
        d[cat_cols] = enc.transform(d[cat_cols].astype(str))
    medians = {}
    for c in num_cols:
        d[c] = d[c].astype(float).replace([np.inf, -np.inf], np.nan)
        medians[c] = d.loc[train_mask, c].median()
        d[c] = d[c].fillna(medians[c])
    all_cols = num_cols + cat_cols
    scaler = StandardScaler().fit(d.loc[train_mask, all_cols].to_numpy())
    X = scaler.transform(d[all_cols].to_numpy()).astype(np.float32)

    b = Base()
    b.df = df
    b.X = X
    b.cols = all_cols
    b.n_num = len(num_cols)
    b.mean = scaler.mean_.astype(np.float64)
    b.scale = scaler.scale_.astype(np.float64)
    # standardized value used when a field is missing: train median (numeric) / unknown code (categorical)
    fill_raw = np.array([medians[c] for c in num_cols] + [unknown_code] * len(cat_cols), dtype=np.float64)
    b.fill_std = ((fill_raw - b.mean) / b.scale).astype(np.float32)
    le = LabelEncoder()
    b.y = le.fit_transform(df["Attack_type"].astype(str))
    b.classes = le.classes_
    b.n_classes = len(le.classes_)
    b.split_pos = {s: np.flatnonzero((df["split"] == s).to_numpy()) for s in ["train", "val", "test"]}
    b.flow_idx = [all_cols.index(c) for c in FLOW_COLS if c in all_cols]
    b.timing_idx = [i for i, c in enumerate(all_cols)
                    if i < b.n_num and any(s in c for s in TIMING_SUBSTR)]
    b.cat_idx = list(range(b.n_num, len(all_cols)))
    b.num_idx = list(range(b.n_num))
    b.fill_raw = fill_raw
    b.cat_cols, b.num_cols = cat_cols, num_cols
    _BASES[variant] = b
    if variant == "default":
        _BASE = b
    return b


# ----------------------------------------------------------------------------------
# Windows, graphs, prototypes
# ----------------------------------------------------------------------------------
def build_windows_fast(pos_stream, node_stream, w=W):
    """Causal per-node window of the w most recent events in stream order (left-padded with the
    first available event). Equivalent to common.build_windows but vectorised."""
    n = len(pos_stream)
    order = np.argsort(node_stream, kind="stable")
    sn = node_stream[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sn)) + 1]
    counts = np.diff(np.r_[starts, n])
    group_start = np.repeat(starts, counts)
    rank = np.arange(n) - group_start
    cols = [group_start + np.maximum(rank - (w - 1 - j), 0) for j in range(w)]
    win_sorted = pos_stream[order][np.stack(cols, axis=1)]
    out = np.empty((n, w), dtype=np.int64)
    out[order] = win_sorted
    return out


def stream_order(base, split, stream):
    pos = base.split_pos[split]
    if stream == "original":
        return pos
    rng = np.random.default_rng({"train": 11, "val": 12, "test": 13}[split])
    return pos[rng.permutation(len(pos))]


def adjacency_from_pairs(src, dst, n):
    counts = np.zeros((n, n), dtype=np.float64)
    np.add.at(counts, (src, dst), 1.0)
    row = counts.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    probs = counts / row            # probs[u, v] = P(u -> v)
    return probs.T.astype(np.float32)  # A_in[v, u]


def fold_hosts(hosts_unique, top_hosts):
    top_set = {h: i for i, h in enumerate(top_hosts)}
    out = {}
    for ip in hosts_unique:
        if ip in top_set:
            out[ip] = top_set[ip]
            continue
        best, best_score = 0, -1
        ip_parts = str(ip).split(".")
        for h in top_hosts:
            score = 0
            for a, b in zip(ip_parts, h.split(".")):
                if a == b:
                    score += 1
                else:
                    break
            if score > best_score:
                best_score, best = score, top_set[h]
        out[ip] = best
    return out


_AUX = None


def load_aux():
    global _AUX
    if _AUX is None:
        _AUX = pd.read_parquet(os.path.join(DATA_DIR, "aux_hosts.parquet"))
    return _AUX


def dst_nodes(base, system):
    aux = load_aux()
    tr = aux.loc[base.split_pos["train"], "ip.src_host"]
    top_hosts = tr.value_counts().head(system).index.tolist()
    m = fold_hosts(aux["ip.dst_host"].unique(), top_hosts)
    # alignment self-check: the source-side fold must reproduce the stored node column
    ms = fold_hosts(aux["ip.src_host"].unique(), top_hosts)
    src_nodes = aux["ip.src_host"].map(ms).to_numpy()
    assert (src_nodes == base.df[f"node_{system}"].to_numpy()).all(), "src node reconstruction mismatch"
    return aux["ip.dst_host"].map(m).to_numpy().astype(np.int64)


def build_adjacency(base, system, graph, stream, graph_seed=0):
    node = base.df[f"node_{system}"].to_numpy().astype(np.int64)
    tr_stream = stream_order(base, "train", stream)
    seq = node[tr_stream]
    if graph == "transition":
        return adjacency_from_pairs(seq[:-1], seq[1:], system)
    if graph == "block_shuffle":
        rng = np.random.default_rng(graph_seed)
        blocks = [seq[i:i + 200] for i in range(0, len(seq), 200)]
        seq2 = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        return adjacency_from_pairs(seq2[:-1], seq2[1:], system)
    if graph == "shuffle":
        rng = np.random.default_rng(graph_seed)
        seq2 = seq[rng.permutation(len(seq))]
        return adjacency_from_pairs(seq2[:-1], seq2[1:], system)
    if graph == "link":
        dn = dst_nodes(base, system)
        return adjacency_from_pairs(node[tr_stream], dn[tr_stream], system)
    if graph == "rewired":
        A = adjacency_from_pairs(seq[:-1], seq[1:], system)
        rng = np.random.default_rng(graph_seed)
        A2 = np.zeros_like(A)
        for u in range(system):          # permute destinations of each source's out-edges
            A2[:, u] = A[rng.permutation(system), u]
        return A2
    raise ValueError(graph)


def compute_prototypes(Xtr, node_tr, n):
    d = Xtr.shape[1]
    proto = np.zeros((n, d), dtype=np.float64)
    cnt = np.bincount(node_tr, minlength=n)
    np.add.at(proto, node_tr, Xtr)
    glob = Xtr.mean(axis=0)
    ok = cnt >= N_MIN_PROTO
    proto[ok] /= cnt[ok, None]
    proto[~ok] = glob
    return proto.astype(np.float32), cnt


def structural_traffic_weighted(A_in, node_weights):
    """Same four descriptors as common.structural_node_features, but each channel is standardized
    with the traffic-weighted mean/std (weights = per-node training event counts) instead of the
    uniform mean/std over nodes, so descriptor scales do not depend on how traffic is spread
    over the nodes of a particular graph."""
    import networkx as nx
    n = A_in.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for v in range(n):
        for u in range(n):
            if A_in[v, u] > 0:
                G.add_edge(u, v, weight=float(A_in[v, u]))
    indeg = np.array([G.in_degree(v, weight="weight") for v in range(n)], dtype=np.float64)
    outdeg = np.array([G.out_degree(v, weight="weight") for v in range(n)], dtype=np.float64)
    pr = nx.pagerank(G, weight="weight")
    pagerank = np.array([pr.get(v, 0.0) for v in range(n)])
    clus = nx.clustering(G.to_undirected(), weight="weight")
    clustering = np.array([clus.get(v, 0.0) for v in range(n)])
    f = np.stack([indeg, outdeg, pagerank, clustering], axis=1)
    w = np.asarray(node_weights, dtype=np.float64)
    w = w / w.sum()
    mu = (w[:, None] * f).sum(0)
    sd = np.sqrt((w[:, None] * (f - mu) ** 2).sum(0))
    return ((f - mu) / (sd + 1e-6)).astype(np.float32)


class GraphCtx:
    def __init__(self, A_in, proto, k=K, node_weights=None):
        self.A_in = A_in.astype(np.float32)
        self.n = A_in.shape[0]
        A_out = A_in.T.copy()
        sf = (structural_node_features(A_in) if node_weights is None
              else structural_traffic_weighted(A_in, node_weights))
        self.struct = to_t(sf).to(DEVICE)
        ii, iw = topk_neighbors(A_in, k)
        oi, ow = topk_neighbors(A_out, k)
        self.in_idx, self.in_w = to_t(ii, torch.long).to(DEVICE), to_t(iw).to(DEVICE)
        self.out_idx, self.out_w = to_t(oi, torch.long).to(DEVICE), to_t(ow).to(DEVICE)
        self.in_m, self.out_m = self.in_w > 0, self.out_w > 0
        self.proto = to_t(proto).to(DEVICE)
        self.neigh_in = to_t(A_in @ proto).to(DEVICE)       # [n, F] dense weighted aggregation (baselines)
        self.neigh_out = to_t(A_out @ proto).to(DEVICE)
        self.n_edges = int((A_in > 0).sum())
        self.frac_nodes_lt_k_in = float(((A_in > 0).sum(1) < k).mean())
        self.frac_nodes_lt_k_out = float(((A_in > 0).sum(0) < k).mean())
        self.n_isolated = int((((A_in > 0).sum(1) == 0) & ((A_in > 0).sum(0) == 0)).sum())


class Split:
    pass


def build_dataset(base, system=100, graph="transition", stream="original", graph_seed=0, struct_norm="node"):
    node_all = base.df[f"node_{system}"].to_numpy().astype(np.int64)
    ds = {}
    for s in ["train", "val", "test"]:
        pos = stream_order(base, s, stream)
        sp = Split()
        sp.pos = pos
        sp.node = to_t(node_all[pos], torch.long).to(DEVICE)
        sp.node_np = node_all[pos]
        sp.win = to_t(build_windows_fast(pos, node_all[pos]), torch.long).to(DEVICE)
        sp.y_np = base.y[pos]
        sp.y = to_t(sp.y_np, torch.long).to(DEVICE)
        sp.n = len(pos)
        ds[s] = sp
    A_in = build_adjacency(base, system, graph, stream, graph_seed)
    tr_pos = ds["train"].pos
    proto, cnt = compute_prototypes(base.X[tr_pos], ds["train"].node_np, system)
    ctx = GraphCtx(A_in, proto, node_weights=(cnt if struct_norm == "traffic" else None))
    ctx.node_counts = cnt
    return dict(splits=ds, ctx=ctx, system=system, graph=graph, stream=stream)


def flat_split(base, ds_split, X=None):
    X = base.X if X is None else X
    return X[ds_split.pos]


# ----------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------
class MaskedSparseAttn(nn.Module):
    """Learned attention over each node's top-k neighbours; zero-weight slots are masked."""
    def __init__(self, feat_dim, hidden):
        super().__init__()
        self.wq = nn.Linear(hidden, hidden)
        self.wk = nn.Linear(feat_dim, hidden)
        self.wv = nn.Linear(feat_dim, hidden)
        self.edge_bias = nn.Linear(1, 1)
        self.out_dim = hidden

    def forward(self, query, nf, ew, em):
        q = self.wq(query).unsqueeze(1)
        k = self.wk(nf)
        v = self.wv(nf)
        scores = (q * k).sum(-1) / (k.shape[-1] ** 0.5) + self.edge_bias(ew.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(~em, -1e9)
        attn = torch.softmax(scores, dim=1) * em.float()
        return (attn.unsqueeze(-1) * v).sum(dim=1)


class MaskedDenseAgg(DenseNeighborAgg):
    def forward(self, query, nf, ew, em):
        w = ew * em.float()
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        return self.proj((nf * w.unsqueeze(-1)).sum(dim=1))


class SBGAFv2(nn.Module):
    def __init__(self, in_dim, struct_dim, n_classes, hidden=192, layers=2, dropout=0.3397475692650171,
                 bidir=True, struct=True, graph=True, sparse=True, temporal="lstm_attn", in_branch=True):
        super().__init__()
        self.bidir, self.struct, self.graph = bidir and graph, struct and graph, graph
        self.in_branch = in_branch and graph
        self.temporal = (TCNEncoder(in_dim, hidden, layers, dropout, "relu") if temporal == "tcn"
                         else LSTMAttnEncoder(in_dim, hidden, layers, dropout, "relu"))
        dim = self.temporal.out_dim
        if self.struct:
            self.struct_proj = nn.Linear(struct_dim, hidden)
            dim += hidden
        cls = MaskedSparseAttn if sparse else MaskedDenseAgg
        if self.in_branch:
            self.attn_in = cls(in_dim, hidden)
            dim += hidden
        if self.bidir:
            self.attn_out = cls(in_dim, hidden)
            dim += hidden
        self.dense = nn.Sequential(nn.Linear(dim, hidden), act_layer("relu"), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, xw, sfeat, fi, wi, mi, fo, wo, mo):
        h = self.temporal(xw)
        parts = [h]
        if self.struct:
            parts.append(self.struct_proj(sfeat))
        if self.in_branch:
            parts.append(self.attn_in(h, fi, wi, mi))
        if self.bidir:
            parts.append(self.attn_out(h, fo, wo, mo))
        return self.out(self.dense(torch.cat(parts, dim=1)))


class NodeEmbLSTM(nn.Module):
    """Node-indexed baseline: the SBGA-F temporal encoder plus a learned embedding per node id.
    Its parameter count depends on the number of nodes, so it cannot be moved to another graph."""
    def __init__(self, in_dim, n_nodes, n_classes, hidden=192, layers=2, dropout=0.3397475692650171):
        super().__init__()
        self.n_nodes = n_nodes
        self.temporal = LSTMAttnEncoder(in_dim, hidden, layers, dropout, "relu")
        self.emb = nn.Embedding(n_nodes, hidden)
        self.dense = nn.Sequential(nn.Linear(self.temporal.out_dim + hidden, hidden), act_layer("relu"), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, xw, nid):
        return self.out(self.dense(torch.cat([self.temporal(xw), self.emb(nid % self.n_nodes)], dim=1)))


ARMS = {
    "Full":              dict(),
    "no_bidirectional":  dict(bidir=False),
    "no_structural":     dict(struct=False),
    "dense_aggregation": dict(sparse=False),
    "TCN_encoder":       dict(temporal="tcn"),
    "CE_loss":           dict(loss="ce"),
    "no_graph":          dict(graph=False),
    "out_only":          dict(in_branch=False),
}


class Predictor:
    """Wraps a trained model with the closure that turns (split, idx) into logits."""
    def __init__(self, kind, model, base):
        self.kind, self.model, self.base = kind, model, base

    def logits(self, ds_split, ctx, X_gpu, idx):
        if self.kind == "sbgaf":
            nid = ds_split.node[idx]
            return self.model(X_gpu[ds_split.win[idx]], ctx.struct[nid],
                              ctx.proto[ctx.in_idx[nid]], ctx.in_w[nid], ctx.in_m[nid],
                              ctx.proto[ctx.out_idx[nid]], ctx.out_w[nid], ctx.out_m[nid])
        if self.kind == "nodeemb":
            return self.model(X_gpu[ds_split.win[idx]], ds_split.node[idx])
        if self.kind.startswith("ids:"):
            graph = MODEL_SPECS[self.kind[4:]]["graph"]
            xb = X_gpu[ds_split.win[idx]]
            if graph == "none":
                return self.model(xb)
            nid = ds_split.node[idx]
            return self.model(xb, ctx.neigh_in[nid], ctx.neigh_out[nid] if graph == "bi" else None)
        if self.kind == "ft":
            return self.model(X_gpu[ds_split.pos_t[idx]])
        raise ValueError(self.kind)

    @torch.no_grad()
    def proba(self, ds_split, ctx, X_gpu, chunk=16384):
        self.model.eval()
        out = []
        for i in range(0, ds_split.n, chunk):
            idx = torch.arange(i, min(i + chunk, ds_split.n), device=DEVICE)
            out.append(F.softmax(self.logits(ds_split, ctx, X_gpu, idx), dim=1))
        return torch.cat(out).cpu().numpy()


def macro_f1(y, p):
    return f1_score(y, p, average="macro", zero_division=0) * 100


def min_recall(y, p, n_classes=15):
    return float(np.min(recall_score(y, p, average=None, labels=list(range(n_classes)), zero_division=0))) * 100


def _fit_loop(pred, data, X_gpu, seed, loss_fn, lr, batch, max_epochs, patience):
    ds, ctx = data["splits"], data["ctx"]
    tr, va = ds["train"], ds["val"]
    model = pred.model
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_f1, best_state, bad = -1.0, None, 0
    if SMOKE:
        max_epochs = 1
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(tr.n, device=DEVICE)
        for i in range(0, tr.n, batch):
            idx = perm[i:i + batch]
            loss = loss_fn(pred.logits(tr, ctx, X_gpu, idx), tr.y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        vp = pred.proba(va, ctx, X_gpu).argmax(1)
        vf1 = f1_score(va.y_np, vp, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return time.time() - t0


def train_sbgaf(base, data, X_gpu, seed, arm="Full"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = dict(ARMS[arm])
    loss_kind = cfg.pop("loss", "focal")
    model = SBGAFv2(base.X.shape[1], 4, base.n_classes, HP["hidden"], HP["layers"], HP["dropout"], **cfg).to(DEVICE)
    cw = class_weights_sqrt(data["splits"]["train"].y_np, base.n_classes)
    if loss_kind == "focal":
        loss_fn = FocalLoss(cw, gamma=HP["gamma"]).to(DEVICE)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=to_t(cw).to(DEVICE))
    pred = Predictor("sbgaf", model, base)
    cud = torch.backends.cudnn.enabled
    if cfg.get("temporal") == "tcn":      # convolutions need cuDNN; only the LSTM kernel is switched off
        torch.backends.cudnn.enabled = True
    t = _fit_loop(pred, data, X_gpu, seed, loss_fn, HP["lr"], HP["batch"], HP["max_epochs"], HP["patience"])
    torch.backends.cudnn.enabled = cud
    pred.train_time = t
    pred.n_params = sum(p.numel() for p in model.parameters())
    return pred


def train_nodeemb(base, data, X_gpu, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = NodeEmbLSTM(base.X.shape[1], data["system"], base.n_classes, HP["hidden"], HP["layers"], HP["dropout"]).to(DEVICE)
    cw = class_weights_sqrt(data["splits"]["train"].y_np, base.n_classes)
    pred = Predictor("nodeemb", model, base)
    pred.train_time = _fit_loop(pred, data, X_gpu, seed, FocalLoss(cw, gamma=HP["gamma"]).to(DEVICE), HP["lr"], HP["batch"],
                                HP["max_epochs"], HP["patience"])
    pred.n_params = sum(p.numel() for p in model.parameters())
    return pred


def train_ids(base, data, X_gpu, name, seed, hp=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    spec = MODEL_SPECS[name]
    hp = dict(dict(hidden=128, layers=3, dropout=0.2, lr=1e-3), **(hp or {}))
    model = IDSModel(base.X.shape[1], hp["hidden"], hp["layers"], hp["dropout"], "relu", temporal=spec["temporal"],
                     graph=spec["graph"], n_classes=base.n_classes).to(DEVICE)
    cw = to_t(class_weights_sqrt(data["splits"]["train"].y_np, base.n_classes)).to(DEVICE)
    pred = Predictor(f"ids:{name}", model, base)
    t = _fit_loop(pred, data, X_gpu, seed, nn.CrossEntropyLoss(weight=cw), hp["lr"], 1024, 25, 8)
    pred.train_time = t
    pred.n_params = sum(p.numel() for p in model.parameters())
    return pred


def train_ft(base, data, X_gpu, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FTTransformer(n_features=base.X.shape[1], d_token=32, n_layers=3, n_heads=4, dropout=0.1,
                          n_classes=base.n_classes).to(DEVICE)
    cw = to_t(class_weights_sqrt(data["splits"]["train"].y_np, base.n_classes)).to(DEVICE)
    for s in data["splits"].values():
        s.pos_t = to_t(s.pos, torch.long).to(DEVICE)
    pred = Predictor("ft", model, base)
    t = _fit_loop(pred, data, X_gpu, seed, nn.CrossEntropyLoss(weight=cw), 1e-3, 1024, 25, 8)
    pred.train_time = t
    pred.n_params = sum(p.numel() for p in model.parameters())
    return pred


# ----------------------------------------------------------------------------------
# Classical / tree models
# ----------------------------------------------------------------------------------
def make_classical(name, seed, n_classes, params=None):
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    f = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, n_jobs=-1),
        "LinearSVM_SGD": lambda: SGDClassifier(loss="hinge", class_weight="balanced", random_state=seed, max_iter=50),
        "kNN": lambda: KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=200, max_depth=20, class_weight="balanced", random_state=seed, n_jobs=-1),
        "XGBoost": lambda: xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, objective="multi:softprob", num_class=n_classes, random_state=seed, n_jobs=-1, eval_metric="mlogloss"),
        "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05, objective="multiclass", num_class=n_classes, class_weight="balanced", random_state=seed, verbosity=-1),
        "CatBoost": lambda: CatBoostClassifier(iterations=300, depth=8, learning_rate=0.1, loss_function="MultiClass", auto_class_weights="SqrtBalanced", random_seed=seed, verbose=False),
    }
    clf = f[name]()
    if params:
        clf.set_params(**params)
    if SMOKE:
        clf.set_params(**({"iterations": 15} if name == "CatBoost" else {"n_estimators": 15} if name in ("RandomForest", "XGBoost", "LightGBM") else {}))
    return clf


def fit_classical(name, seed, base, Xtr, ytr, params=None):
    clf = make_classical(name, seed, base.n_classes, params)
    t0 = time.time()
    if name == "kNN":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ytr), size=min(30000, len(ytr)), replace=False)
        clf.fit(Xtr[idx], ytr[idx])
    else:
        clf.fit(Xtr, ytr)
    clf.train_time = time.time() - t0
    return clf


def classical_proba(clf, name, X):
    if name == "LinearSVM_SGD":
        s = clf.decision_function(X)
        e = np.exp(s - s.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    return clf.predict_proba(X)


# ----------------------------------------------------------------------------------
# Stacking
# ----------------------------------------------------------------------------------
def fit_meta(p_val_list, y_val):
    from sklearn.linear_model import LogisticRegression
    meta = LogisticRegression(max_iter=2000)
    meta.fit(np.concatenate(p_val_list, axis=1), y_val)
    return meta


# ----------------------------------------------------------------------------------
# Perturbations (applied to TEST rows of the feature matrix)
# ----------------------------------------------------------------------------------
def perturb_features(base, kind, level, seed=0, split="test"):
    X = base.X.copy()
    rows = base.split_pos[split]
    rng = np.random.default_rng(1000 + seed)
    sub = X[rows]
    n, d = sub.shape
    if kind == "num_noise":
        sub[:, base.num_idx] += rng.normal(0, level, size=(n, len(base.num_idx))).astype(np.float32)
    elif kind == "cat_corrupt":
        tr_rows = base.split_pos["train"]
        for j in base.cat_idx:
            flip = rng.random(n) < level
            donor = X[tr_rows[rng.integers(0, len(tr_rows), size=n)], j]
            sub[flip, j] = donor[flip]
    elif kind == "missing":
        m = rng.random((n, d)) < level
        sub = np.where(m, base.fill_std[None, :], sub)
    elif kind == "flow_dropout":
        sub[:, base.flow_idx] = base.fill_std[base.flow_idx][None, :]
    elif kind == "jitter":
        j = base.timing_idx
        raw = sub[:, j] * base.scale[j] + base.mean[j]
        raw = raw * np.exp(rng.normal(0, level, size=raw.shape))
        sub[:, j] = ((raw - base.mean[j]) / base.scale[j]).astype(np.float32)
    elif kind == "combined":
        j = base.timing_idx
        raw = sub[:, j] * base.scale[j] + base.mean[j]
        raw = raw * np.exp(rng.normal(0, 0.25, size=raw.shape))
        sub[:, j] = ((raw - base.mean[j]) / base.scale[j]).astype(np.float32)
        m = rng.random((n, d)) < 0.10
        sub = np.where(m, base.fill_std[None, :], sub)
        tr_rows = base.split_pos["train"]
        for jj in base.cat_idx:
            flip = rng.random(n) < 0.05
            donor = X[tr_rows[rng.integers(0, len(tr_rows), size=n)], jj]
            sub[flip, jj] = donor[flip]
    elif kind == "clean":
        pass
    else:
        raise ValueError(kind)
    X[rows] = sub.astype(np.float32)
    return X


def perturb_adjacency(A_in, kind, level, seed=0):
    rng = np.random.default_rng(2000 + seed)
    A = A_in.copy()
    n = A.shape[0]
    if kind == "edge_drop":
        keep = rng.random(A.shape) >= level
        A = A * keep
    elif kind == "weight_noise":
        A = A * np.exp(rng.normal(0, level, size=A.shape)).astype(np.float32) * (A > 0)
    elif kind == "rewire":
        for u in range(n):
            nz = np.flatnonzero(A[:, u])
            sel = nz[rng.random(len(nz)) < level]
            if len(sel):
                new = rng.choice(n, size=len(sel), replace=False if len(sel) <= n else True)
                w = A[sel, u].copy()
                A[sel, u] = 0
                A[new, u] += w
    elif kind == "clean":
        pass
    else:
        raise ValueError(kind)
    col = A.sum(axis=0, keepdims=True)
    col[col == 0] = 1.0
    return (A / col).astype(np.float32)


# ----------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------
def full_metrics(y, proba, n_classes=15):
    from sklearn.metrics import balanced_accuracy_score, precision_score, roc_auc_score
    pred = proba.argmax(1)
    try:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(range(n_classes)))
    except Exception:
        auc = float("nan")
    rec = recall_score(y, pred, average=None, labels=list(range(n_classes)), zero_division=0)
    return dict(macro_F1=f1_score(y, pred, average="macro", zero_division=0) * 100,
                weighted_F1=f1_score(y, pred, average="weighted", zero_division=0) * 100,
                balanced_acc=balanced_accuracy_score(y, pred) * 100,
                macro_precision=precision_score(y, pred, average="macro", zero_division=0) * 100,
                macro_recall=float(rec.mean()) * 100, macro_auc_ovr=auc,
                min_class_recall=float(rec.min()) * 100)


def save_probs(seed, name, split, p):
    os.makedirs(PROB_DIR, exist_ok=True)
    np.save(os.path.join(PROB_DIR, f"{seed}_{name}_{split}.npy"), p.astype(np.float32))


def load_probs(seed, name, split):
    return np.load(os.path.join(PROB_DIR, f"{seed}_{name}_{split}.npy"))
