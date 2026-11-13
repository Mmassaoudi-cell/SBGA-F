"""Round-2 experiment F: independent dataset. TON_IoT network flows (UNSW testbed, Zeek connection records,
10 classes, 211,043 records). The Edge-IIoTset protocol is applied without re-tuning any hyperparameter:
class-wise 70/10/20 split in file order after de-duplication, identifier columns removed, nodes = the N most
frequent source hosts of the training partition, event-transition graph from the training stream.
Benchmarks: tree ensembles, RNN, GNN, BiRAGNN, SBGA-F, temporal-only SBGA-F, the stacks; cross-topology reuse
from 50 to 25 and 8 nodes.
Usage: python 36_r2_ton.py <seed>..."""
import os, sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
sys.path.insert(0, os.path.dirname(__file__))
import exp_common as E

SRC = r"C:\Users\MMASSAOUDI\Desktop\Data\TONIoT Network Dataset\train_test_network.csv"
OUT = os.path.join(E.RESULTS_DIR, "rev2")
os.makedirs(OUT, exist_ok=True)
ID_COLS = ["src_ip", "dst_ip", "src_port", "dst_port", "dns_query", "ssl_subject", "ssl_issuer", "http_uri",
           "http_user_agent", "http_orig_mime_types", "http_resp_mime_types"]
SIZES = [50, 25, 8]


def load_ton():
    df = pd.read_csv(SRC, low_memory=False)
    n0 = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    df["split"] = "test"
    for _, idx in df.groupby("type", sort=False).indices.items():
        n = len(idx)
        a, b = int(round(n * 0.7)), int(round(n * 0.7)) + int(round(n * 0.1))
        df.loc[idx[:a], "split"] = "train"
        df.loc[idx[a:b], "split"] = "val"
    tr = (df["split"] == "train").to_numpy()
    for k in SIZES:
        top = df.loc[tr, "src_ip"].value_counts().head(k).index.tolist()
        m = E.fold_hosts(df["src_ip"].unique(), top)
        df[f"node_{k}"] = df["src_ip"].map(m).astype(np.int64)
    cols = [c for c in df.columns if c not in ID_COLS + ["label", "type", "split"] and not c.startswith("node_")]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if c not in num]
    d = df[cols].copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(d.loc[tr, cat].astype(str))
    d[cat] = enc.transform(d[cat].astype(str))
    med = {c: d.loc[tr, c].median() for c in num}
    allc = num + cat
    sc = StandardScaler().fit(d.loc[tr, allc].to_numpy())
    b = E.Base()
    b.df, b.cols, b.n_num = df, allc, len(num)
    b.X = sc.transform(d[allc].to_numpy()).astype(np.float32)
    b.mean, b.scale = sc.mean_, sc.scale_
    b.fill_std = ((np.array([med[c] for c in num] + [-1] * len(cat)) - b.mean) / b.scale).astype(np.float32)
    le = LabelEncoder()
    b.y = le.fit_transform(df["type"].astype(str))
    b.classes, b.n_classes = le.classes_, len(le.classes_)
    b.split_pos = {s: np.flatnonzero((df["split"] == s).to_numpy()) for s in ["train", "val", "test"]}
    b.num_idx, b.cat_idx = list(range(len(num))), list(range(len(num), len(allc)))
    b.n_raw, b.num_cols, b.cat_cols = n0, num, cat
    return b


def main():
    seeds = [int(s) for s in sys.argv[1:]]
    base = load_ton()
    nc = base.n_classes
    X = torch.as_tensor(base.X).to(E.DEVICE)
    data = {k: E.build_dataset(base, k) for k in SIZES}
    d = data[50]
    tr, va, te = (d["splits"][s] for s in ("train", "val", "test"))
    Xtr, Xva, Xte = base.X[tr.pos], base.X[va.pos], base.X[te.pos]
    if not os.path.exists(os.path.join(OUT, "ton_stats.csv")):
        st = pd.DataFrame({"train": pd.Series(tr.y_np).value_counts(), "val": pd.Series(va.y_np).value_counts(),
                           "test": pd.Series(te.y_np).value_counts()}).sort_index()
        st.index = base.classes[st.index]
        st.to_csv(os.path.join(OUT, "ton_stats.csv"))
        win_y = base.y[te.win.cpu().numpy()]
        info = dict(raw=base.n_raw, dedup=len(base.df), n_features=base.X.shape[1], n_num=base.n_num,
                    n_cat=len(base.cat_idx), src_hosts=base.df.src_ip.nunique(), n_edges_50=d["ctx"].n_edges,
                    window_purity_test=float((win_y == te.y_np[:, None]).all(1).mean()))
        pd.Series(info).to_csv(os.path.join(OUT, "ton_info.csv"))
        print(info, flush=True)
    for seed in seeds:
        path = os.path.join(OUT, f"ton_{seed}.csv")
        if os.path.exists(path):
            continue
        rows, P, trows = [], {}, []

        def rec(name, pv, pt):
            P[name] = (pv, pt)
            np.save(os.path.join(E.PROB_DIR, f"ton_{seed}_{name}_test.npy"), pt.astype(np.float32))
            m = E.full_metrics(te.y_np, pt, nc)
            m.update(model=name, seed=seed)
            rows.append(m)
            print(f"seed {seed} TON {name:22s} macroF1={m['macro_F1']:.2f} minRec={m['min_class_recall']:.1f}", flush=True)

        def transfer(label, pred):
            for k in SIZES:
                t = data[k]["splits"]["test"]
                m = E.full_metrics(t.y_np, pred.proba(t, data[k]["ctx"], X), nc)
                m.update(model=label, seed=seed, target=k)
                trows.append(m)

        for name in ["CatBoost", "LightGBM", "XGBoost", "RandomForest"]:
            clf = E.fit_classical(name, seed, base, Xtr, tr.y_np)
            rec(name, clf.predict_proba(Xva), clf.predict_proba(Xte))
        for name in ["RNN", "GNN", "BiRAGNN"]:
            pred = E.train_ids(base, d, X, name, seed)
            rec(name, pred.proba(va, d["ctx"], X), pred.proba(te, d["ctx"], X))
            if name != "GNN":
                transfer(name, pred)
        for arm, name in [("Full", "SBGA-F"), ("no_graph", "SBGA-F temporal-only")]:
            pred = E.train_sbgaf(base, d, X, seed, arm)
            rec(name, pred.proba(va, d["ctx"], X), pred.proba(te, d["ctx"], X))
            transfer(name, pred)
        for k in (25, 8):
            pred = E.train_sbgaf(base, data[k], X, seed, "Full")
            t = data[k]["splits"]["test"]
            m = E.full_metrics(t.y_np, pred.proba(t, data[k]["ctx"], X), nc)
            m.update(model="SBGA-F retrained on target", seed=seed, target=k)
            trows.append(m)
        for a, label in [("SBGA-F", "SBGA-F-Stack"), ("RNN", "RNN + CatBoost"), ("BiRAGNN", "BiRAGNN + CatBoost"),
                         ("SBGA-F temporal-only", "Temporal-only + CatBoost"), ("LightGBM", "LightGBM + CatBoost")]:
            meta = E.fit_meta([P[a][0], P["CatBoost"][0]], va.y_np)
            rec(label, meta.predict_proba(np.concatenate([P[a][0], P["CatBoost"][0]], axis=1)),
                meta.predict_proba(np.concatenate([P[a][1], P["CatBoost"][1]], axis=1)))
        avg = (P["SBGA-F"][1] + P["CatBoost"][1]) / 2
        rec("SBGA-F + CatBoost, average", (P["SBGA-F"][0] + P["CatBoost"][0]) / 2, avg)
        pd.DataFrame(rows).to_csv(path, index=False)
        pd.DataFrame(trows).to_csv(os.path.join(OUT, f"ton_transfer_{seed}.csv"), index=False)


if __name__ == "__main__":
    main()
