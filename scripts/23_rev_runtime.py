"""
Revision experiment 5: end-to-end inference-cost profile. Times every stage between raw packet
records and the final class decision on the 59,481-event test partition: field encoding and scaling,
flow-statistics aggregation, window preparation, SBGA-F forward pass, CatBoost prediction, stacking
meta-learner, and (per topology change) adjacency / descriptor / prototype refresh.
Must be run on an otherwise idle machine.  Usage: python 23_rev_runtime.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import exp_common as E

OUT = os.path.join(E.RESULTS_DIR, "rev")
SEED = 42


def timeit(fn, repeats=3):
    best = 1e9
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best, out


def parse_time(s):
    t = s.astype(str).str.strip().str.split(" ").str[-1].str.split(":", expand=True)
    for c in (0, 1, 2):  # malformed raw rows become NaN, as in 01_prepare_data.py
        t[c] = pd.to_numeric(t[c], errors="coerce") if c in t.columns else np.nan
    return t[0] * 3600 + t[1] * 60 + t[2]


def flow_aggregate(df_rows, aux_rows):
    """Flow statistics (8 aggregates) over host-pair flows chunked to 25 packets, computed from raw
    packet records of one partition (deployment form: the key uses host pair only)."""
    d = pd.DataFrame({"src": aux_rows["ip.src_host"].to_numpy(), "dst": aux_rows["ip.dst_host"].to_numpy(),
                      "t": parse_time(aux_rows["frame.time"]).to_numpy()})
    d["len"] = df_rows[["tcp.len", "http.content_length", "mqtt.len"]].fillna(0).max(axis=1).to_numpy()
    base_key = d["src"] + "|" + d["dst"]
    chunk = base_key + "|" + (d.groupby(base_key).cumcount() // 25).astype(str)
    g = d.groupby(chunk)
    agg = g.agg(start=("t", "min"), end=("t", "max"), total_packets=("t", "count"), total_bytes=("len", "sum"),
                max_packet_length=("len", "max"))
    agg["flow_duration"] = (agg["end"] - agg["start"]).clip(lower=0)
    agg["pps"] = agg["total_packets"] / agg["flow_duration"].replace(0, np.nan)
    iat = d.sort_values("t").groupby(chunk.loc[d.sort_values("t").index])["t"].diff()
    ia = iat.groupby(chunk.loc[iat.index]).agg(["mean", "min", "max"])
    return agg.join(ia)


def main():
    base = E.load_base()
    aux = E.load_aux()
    data = E.build_dataset(base, 100)
    X = torch.as_tensor(base.X).to(E.DEVICE)
    tr, va, te = (data["splits"][s] for s in ("train", "val", "test"))
    ctx = data["ctx"]
    n_te = te.n
    rows = []

    # ---- models
    sb = E.train_sbgaf(base, data, X, SEED, "Full")
    cat = E.fit_classical("CatBoost", SEED, base, E.flat_split(base, tr), tr.y_np)
    pv_sb, pv_cb = sb.proba(va, ctx, X), cat.predict_proba(E.flat_split(base, va))
    meta = E.fit_meta([pv_sb, pv_cb], va.y_np)
    bir = E.train_ids(base, data, X, "BiRAGNN", SEED)

    # ---- stage 1: field encoding + scaling of raw records
    cols = base.cols
    num_cols, cat_cols = cols[:base.n_num], cols[base.n_num:]
    raw = base.df.loc[base.split_pos["test"], cols]
    raw_tr = base.df.loc[base.split_pos["train"], cat_cols].astype(str)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(raw_tr)
    scaler = StandardScaler().fit(np.zeros((2, len(cols))))
    scaler.mean_, scaler.scale_ = base.mean, base.scale

    def encode():
        d = raw.copy()
        d[cat_cols] = enc.transform(d[cat_cols].astype(str))
        return scaler.transform(d.to_numpy(dtype=float)).astype(np.float32)
    t_enc, _ = timeit(encode)

    # ---- stage 2: flow aggregation
    t_flow, _ = timeit(lambda: flow_aggregate(base.df.loc[base.split_pos["test"]], aux.loc[base.split_pos["test"]]), repeats=2)

    # ---- stage 3: window preparation
    node_all = base.df["node_100"].to_numpy().astype(np.int64)
    t_win, _ = timeit(lambda: E.build_windows_fast(te.pos, node_all[te.pos]))

    # ---- stage 4: SBGA-F forward (GPU, batch 4096) and BiRAGNN
    t_sb, _ = timeit(lambda: sb.proba(te, ctx, X, chunk=4096))
    t_bir, _ = timeit(lambda: bir.proba(te, ctx, X, chunk=4096))

    # ---- stage 5: CatBoost + meta-learner
    Xte = E.flat_split(base, te)
    t_cat, p_cb = timeit(lambda: cat.predict_proba(Xte))
    p_sb = sb.proba(te, ctx, X)
    t_meta, _ = timeit(lambda: meta.predict_proba(np.concatenate([p_sb, p_cb], axis=1)))

    # ---- stage 6: topology refresh (adjacency, descriptors, top-k, prototypes), per topology change
    def refresh():
        A = E.build_adjacency(base, 100, "transition", "original")
        proto, cnt = E.compute_prototypes(base.X[tr.pos], tr.node_np, 100)
        return E.GraphCtx(A, proto)
    t_refresh, _ = timeit(refresh, repeats=3)

    # ---- single-event latency (batch size 1)
    def single_sb(n=200):
        t0 = time.perf_counter()
        for i in range(n):
            idx = torch.tensor([i], device=E.DEVICE)
            with torch.no_grad():
                sb.logits(te, ctx, X, idx)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n
    single_sb(20)
    lat_sb = single_sb()

    def single_cat(n=200):
        t0 = time.perf_counter()
        for i in range(n):
            cat.predict_proba(Xte[i:i + 1])
        return (time.perf_counter() - t0) / n
    lat_cat = single_cat()

    # ---- CPU inference of SBGA-F (edge-style deployment), 1 socket threads
    torch.set_num_threads(8)
    E.DEVICE = torch.device("cpu")
    import exp_common as E2
    E2.DEVICE = torch.device("cpu")
    sb_cpu = E.Predictor("sbgaf", sb.model.to("cpu"), base)
    data_cpu = E.build_dataset(base, 100)
    X_cpu = torch.as_tensor(base.X)
    sub = data_cpu["splits"]["test"]
    sub_n = 10000
    sub.n = sub_n
    t_cpu, _ = timeit(lambda: sb_cpu.proba(sub, data_cpu["ctx"], X_cpu, chunk=1024), repeats=2)
    cpu_ms = t_cpu / sub_n * 1000

    ms = lambda t: t / n_te * 1000
    rows = [
        dict(stage="Field encoding and scaling", total_s=t_enc, ms_per_event=ms(t_enc)),
        dict(stage="Flow-statistics aggregation", total_s=t_flow, ms_per_event=ms(t_flow)),
        dict(stage="Window preparation", total_s=t_win, ms_per_event=ms(t_win)),
        dict(stage="SBGA-F forward (GPU, batch 4096)", total_s=t_sb, ms_per_event=ms(t_sb)),
        dict(stage="BiRAGNN forward (GPU, batch 4096)", total_s=t_bir, ms_per_event=ms(t_bir)),
        dict(stage="CatBoost prediction (CPU)", total_s=t_cat, ms_per_event=ms(t_cat)),
        dict(stage="Stacking meta-learner", total_s=t_meta, ms_per_event=ms(t_meta)),
        dict(stage="Topology refresh (per change, 100 nodes)", total_s=t_refresh, ms_per_event=float("nan")),
        dict(stage="SBGA-F single-event latency (GPU, batch 1)", total_s=lat_sb, ms_per_event=lat_sb * 1000),
        dict(stage="CatBoost single-event latency (CPU)", total_s=lat_cat, ms_per_event=lat_cat * 1000),
        dict(stage="SBGA-F forward (CPU, 8 threads, batch 1024)", total_s=t_cpu, ms_per_event=cpu_ms),
    ]
    pre = ms(t_enc) + ms(t_flow)
    e2e = {
        "CatBoost": pre + ms(t_cat),
        "SBGA-F": pre + ms(t_win) + ms(t_sb),
        "BiRAGNN": pre + ms(t_win) + ms(t_bir),
        "SBGA-F-Stack": pre + ms(t_win) + ms(t_sb) + ms(t_cat) + ms(t_meta),
    }
    for k, v in e2e.items():
        rows.append(dict(stage=f"END-TO-END {k}", total_s=v * n_te / 1000, ms_per_event=v))
    out = pd.DataFrame(rows)
    out["events_per_s"] = 1000.0 / out["ms_per_event"]
    out.to_csv(os.path.join(OUT, "runtime.csv"), index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
