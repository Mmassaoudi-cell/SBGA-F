"""TON_IoT audit (no training): node-identity shortcut, per-class recall of the main models,
and which Edge-IIoTset classes contain test events from unseen source hosts."""
import importlib, os, sys, glob
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
sys.path.insert(0, os.path.dirname(__file__))
import exp_common as E
ton = importlib.import_module("36_r2_ton")
OUT = os.path.join(E.RESULTS_DIR, "rev2")
b = ton.load_ton()
pos = b.split_pos
ytr, yte = b.y[pos["train"]], b.y[pos["test"]]
rows = []
for k in ton.SIZES:
    node = b.df[f"node_{k}"].to_numpy()
    tab = pd.crosstab(node[pos["train"]], ytr)
    pred = pd.Series(node[pos["test"]]).map(tab.idxmax(1)).fillna(pd.Series(ytr).mode()[0]).astype(int).to_numpy()
    rec = recall_score(yte, pred, average=None, labels=list(range(b.n_classes)), zero_division=0)
    rows.append(dict(nodes=k, node_only_macro_F1=f1_score(yte, pred, average="macro", zero_division=0) * 100,
                     node_only_accuracy=(pred == yte).mean() * 100, nodes_one_class=int((tab.gt(0).sum(1) == 1).sum()),
                     nodes_with_events=len(tab), classes_with_zero_recall=int((rec == 0).sum())))
print(pd.DataFrame(rows).round(2).to_string(index=False))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "audit_ton_node_shortcut.csv"), index=False)
src = b.df["src_ip"].to_numpy()
unseen = ~np.isin(src[pos["test"]], np.unique(src[pos["train"]]))
print("TON test events from unseen source hosts: %.2f%%" % (unseen.mean() * 100))
g = b.df.groupby("type")
print(pd.DataFrame({"src_hosts": g.src_ip.nunique(), "top_host_share": g.src_ip.agg(lambda s: s.value_counts(normalize=True).iloc[0]).round(3)}).T.to_string())
# share of each class's test events emitted by hosts that also emit normal traffic in training
norm_hosts = set(b.df[(b.df.split == "train") & (b.df.type == "normal")].src_ip)
te = b.df[b.df.split == "test"]
print("share of test events from hosts that emit normal traffic in training, per class:")
print(te.groupby("type").src_ip.apply(lambda s: s.isin(norm_hosts).mean()).round(3).to_string())
rec_rows = []
for m in ("CatBoost", "RNN", "SBGA-F temporal-only", "BiRAGNN", "SBGA-F", "SBGA-F-Stack"):
    R = []
    for f in glob.glob(os.path.join(E.PROB_DIR, f"ton_*_{m}_test.npy")):
        R.append(recall_score(yte, np.load(f).argmax(1), average=None, labels=list(range(b.n_classes)), zero_division=0) * 100)
    rec_rows.append(pd.Series(np.mean(R, 0), index=b.classes, name=m))
pc = pd.DataFrame(rec_rows).round(1)
print(pc.to_string())
pc.to_csv(os.path.join(OUT, "audit_ton_per_class_recall.csv"))
# Edge-IIoTset: classes of test events from unseen hosts
e = E.load_base(); aux = E.load_aux()
s = aux["ip.src_host"].to_numpy(); p = e.split_pos
u = ~np.isin(s[p["test"]], np.unique(s[p["train"]]))
print("Edge unseen-host test events by class:", pd.Series(e.classes[e.y[p["test"]][u]]).value_counts().to_dict())
