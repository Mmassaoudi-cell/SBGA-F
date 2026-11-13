"""
Re-derives the (deterministic) sampled event table of 01_prepare_data.py and stores the
host-level columns that were dropped from the modeling tables (ip.src_host, ip.dst_host,
frame.time), aligned row-for-row with data/fusion.parquet. Used for the observed
communication-link graph and for the end-to-end runtime profile. Alignment is verified
against the stored Attack_type / split / node_* columns before anything is written.
"""
import os, sys, importlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
prep = importlib.import_module("01_prepare_data")

df = prep.load_raw()
df = prep.clean(df)
df, _ = prep.stratified_temporal_sample(df)
for k in prep.SYSTEM_SIZES:
    df[f"node_{k}"] = prep.assign_nodes(df, k)
df = df.sort_index()

fus = pd.read_parquet(os.path.join(prep.OUT_DIR, "fusion.parquet"))
assert len(fus) == len(df), (len(fus), len(df))
for c in ["Attack_type", "split", "node_8", "node_50", "node_100"]:
    assert (fus[c].to_numpy() == df[c].to_numpy()).all(), f"misaligned on {c}"
aux = df[["ip.src_host", "ip.dst_host", "frame.time"]].reset_index(drop=True)
aux["orig_index"] = df.index.to_numpy()
aux.to_parquet(os.path.join(prep.OUT_DIR, "aux_hosts.parquet"), index=False)
print("aligned and saved aux_hosts.parquet", aux.shape)
