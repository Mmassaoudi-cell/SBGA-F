"""Round-2 audit (no training): leakage diagnostics, topology mappings, stacking controls and
test-set uncertainty computed from the data tables and the stored per-seed probabilities.
Writes results/rev2/audit_*.csv. Usage: python 37_r2_audit.py"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
sys.path.insert(0, os.path.dirname(__file__))
import exp_common as E

warnings.filterwarnings("ignore")
OUT = os.path.join(E.RESULTS_DIR, "rev2")
os.makedirs(OUT, exist_ok=True)
SEEDS = E.SEEDS10
base = E.load_base()
aux = E.load_aux()
NC = base.n_classes
pos = base.split_pos
yte, yva, ytr = base.y[pos["test"]], base.y[pos["val"]], base.y[pos["train"]]


def have(seed, name, split="test"):
    return os.path.exists(os.path.join(E.PROB_DIR, f"{seed}_{name}_{split}.npy"))


def save(name, df):
    df.to_csv(os.path.join(OUT, f"audit_{name}.csv"), index=False)
    print(f"\n== {name}\n{df.round(3).to_string(index=False)}", flush=True)


# 1 ---- window label purity
rows = []
for stream in ("original", "interleaved"):
    for system in (100, 50, 8):
        node = base.df[f"node_{system}"].to_numpy().astype(np.int64)
        for s in ("train", "test"):
            p = E.stream_order(base, s, stream)
            w = E.build_windows_fast(p, node[p])
            same = base.y[w] == base.y[p][:, None]
            rows.append(dict(stream=stream, nodes=system, split=s, pure_windows=same.all(1).mean() * 100,
                             mean_same_label_records=same.mean() * 100))
save("window_purity", pd.DataFrame(rows))

# 2 ---- duplicate feature vectors across partitions
flow = [base.cols.index(c) for c in E.FLOW_COLS]
rows = []
for tag, cols in (("all 54 features", list(range(base.X.shape[1]))),
                  ("46 packet features", [i for i in range(base.X.shape[1]) if i not in flow])):
    Xr = np.ascontiguousarray(base.X[:, cols])
    h = pd.util.hash_pandas_object(pd.DataFrame(Xr), index=False).to_numpy()
    tr = pd.DataFrame({"h": h[pos["train"]], "y": ytr}).groupby("h")["y"].agg(lambda s: set(s))
    te_h = h[pos["test"]]
    in_tr = np.isin(te_h, tr.index.to_numpy())
    same = np.array([yte[i] in tr[te_h[i]] for i in np.flatnonzero(in_tr)])
    per = {base.classes[c]: float(in_tr[yte == c].mean() * 100) for c in range(NC)}
    rows.append(dict(features=tag, test_rows=len(te_h), test_rows_with_identical_train_row=int(in_tr.sum()),
                     share=in_tr.mean() * 100, of_which_same_label=same.mean() * 100 if len(same) else np.nan,
                     unique_test_vectors=len(np.unique(te_h)), per_class=json.dumps({k: round(v, 1) for k, v in per.items()})))
    if tag.startswith("all"):
        dup_mask = in_tr
save("duplicates", pd.DataFrame(rows))

# 3 ---- node-identity shortcut and topology mapping
rows = []
src = aux["ip.src_host"].to_numpy()
for system in (100, 50, 8):
    node = base.df[f"node_{system}"].to_numpy().astype(np.int64)
    tab = pd.crosstab(node[pos["train"]], ytr)
    maj = tab.idxmax(1)
    glob = pd.Series(ytr).mode()[0]
    pred = pd.Series(node[pos["test"]]).map(maj).fillna(glob).astype(int).to_numpy()
    top = pd.Series(src[pos["train"]]).value_counts().head(system).index
    exact = np.isin(src, top)
    rows.append(dict(nodes=system, node_only_macro_F1=f1_score(yte, pred, average="macro", zero_division=0) * 100,
                     node_only_accuracy=(pred == yte).mean() * 100,
                     nodes_with_one_class=int((tab.gt(0).sum(1) == 1).sum()), nodes_with_train_events=len(tab),
                     train_events_at_own_host=exact[pos["train"]].mean() * 100, test_events_at_own_host=exact[pos["test"]].mean() * 100,
                     test_events=len(pos["test"]), distinct_test_hosts=len(np.unique(src[pos["test"]])),
                     test_nodes_used=len(np.unique(node[pos["test"]]))))
n100, n50, n8 = (base.df[f"node_{k}"].to_numpy() for k in (100, 50, 8))
nest = dict(pairs_100_to_50_consistent=float(pd.DataFrame({"a": n100, "b": n50}).groupby("a")["b"].nunique().eq(1).mean() * 100),
            pairs_50_to_8_consistent=float(pd.DataFrame({"a": n50, "b": n8}).groupby("a")["b"].nunique().eq(1).mean() * 100))
save("node_shortcut", pd.DataFrame(rows).assign(**nest))

# 4 ---- probabilities
def P(seed, name, split="test"):
    return E.load_probs(seed, name, split)


seeds = [s for s in SEEDS if have(s, "SBGA-F-Stack")]
unseen_host = ~np.isin(src[pos["test"]], np.unique(src[pos["train"]]))
half = np.zeros(len(yte), bool)                    # second half of every class's test block (later in time)
for c in range(NC):
    i = np.flatnonzero(yte == c)
    half[i[len(i) // 2:]] = True


def sl(y, p, mask):
    cls = [c for c in np.unique(y[mask]) if (y[mask] == c).sum() >= 20]
    return f1_score(y[mask], p[mask], labels=cls, average="macro", zero_division=0) * 100, (y[mask] == p[mask]).mean() * 100, len(cls)


rows = []
for s in seeds:
    for m in ("SBGA-F", "CatBoost", "SBGA-F-Stack", "BiRAGNN", "RNN"):
        pr = P(s, m).argmax(1)
        for tag, mask in (("unseen source hosts", unseen_host), ("seen source hosts", ~unseen_host),
                          ("first half of test block", ~half), ("second half of test block", half),
                          ("identical train vector exists", dup_mask), ("no identical train vector", ~dup_mask)):
            f, a, k = sl(yte, pr, mask)
            rows.append(dict(seed=s, model=m, slice=tag, n=int(mask.sum()), classes=k, macro_F1=f, accuracy=a))
d = pd.DataFrame(rows)
save("slices", d.groupby(["slice", "model"]).agg(n=("n", "first"), classes=("classes", "first"), macro_F1=("macro_F1", "mean"),
                                                 sd=("macro_F1", "std"), accuracy=("accuracy", "mean")).reset_index())
d.to_csv(os.path.join(OUT, "audit_slices_raw.csv"), index=False)

# 5 ---- stacking controls / alternative meta-learners
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb


def lr_stack(vs, ts, yv, **kw):
    m = LogisticRegression(max_iter=2000, **kw).fit(np.concatenate(vs, 1), yv)
    return m.predict_proba(np.concatenate(ts, 1))


rows = []
for i, s in enumerate(seeds):
    s2 = seeds[(i + 1) % len(seeds)]
    sv, st, cv, ct = P(s, "SBGA-F", "val"), P(s, "SBGA-F"), P(s, "CatBoost", "val"), P(s, "CatBoost")
    c2v, c2t = P(s2, "CatBoost", "val"), P(s2, "CatBoost")
    lg = lambda a: np.log(np.clip(a, 1e-6, 1))
    outs = {
        "CatBoost (single)": ct,
        "CatBoost + LR meta-learner (CatBoost only)": lr_stack([cv], [ct], yva),
        "CatBoost + CatBoost (second seed), stacked": lr_stack([cv, c2v], [ct, c2t], yva),
        "SBGA-F + CatBoost, LR meta-learner (proposed)": lr_stack([sv, cv], [st, ct], yva),
        "SBGA-F + CatBoost, LR on log-probabilities": lr_stack([lg(sv), lg(cv)], [lg(st), lg(ct)], yva),
        "SBGA-F + CatBoost, LR with C=0.1": lr_stack([sv, cv], [st, ct], yva, C=0.1),
        "SBGA-F + CatBoost, MLP meta-learner": MLPClassifier((32,), max_iter=300, random_state=s).fit(np.concatenate([sv, cv], 1), yva).predict_proba(np.concatenate([st, ct], 1)),
        "SBGA-F + CatBoost, LightGBM meta-learner": lgb.LGBMClassifier(n_estimators=100, num_leaves=15, learning_rate=0.05, verbosity=-1, random_state=s).fit(np.concatenate([sv, cv], 1), yva).predict_proba(np.concatenate([st, ct], 1)),
    }
    ws = np.linspace(0, 1, 21)
    w = ws[int(np.argmax([f1_score(yva, (a * sv + (1 - a) * cv).argmax(1), average="macro") for a in ws]))]
    outs["SBGA-F + CatBoost, weighted average (weight tuned on validation)"] = w * st + (1 - w) * ct
    outs["SBGA-F + CatBoost, unweighted average"] = (st + ct) / 2
    if have(s, "abl_no_graph"):
        outs["Temporal-only SBGA-F + CatBoost, stacked"] = lr_stack([P(s, "abl_no_graph", "val"), cv], [P(s, "abl_no_graph"), ct], yva)
    ref = outs["SBGA-F + CatBoost, LR meta-learner (proposed)"].argmax(1)
    for k, p in outs.items():
        pr = p.argmax(1)
        rec = recall_score(yte, pr, average=None, labels=list(range(NC)), zero_division=0)
        rows.append(dict(seed=s, config=k, macro_F1=f1_score(yte, pr, average="macro") * 100, min_recall=rec.min() * 100,
                         weight=w if "weighted" in k else np.nan))
d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "audit_stack_controls_raw.csv"), index=False)
save("stack_controls", d.groupby("config", sort=False).agg(macro_F1=("macro_F1", "mean"), sd=("macro_F1", "std"),
                                                             min_recall=("min_recall", "mean"), n=("seed", "count")).reset_index())

# 6 ---- per-class metrics, minimum-recall rules, confusion matrix, test-set uncertainty
rows, mr, raw, mcn = [], [], [], []
cm_pool = {m: np.zeros((NC, NC), np.int64) for m in ("SBGA-F", "CatBoost", "SBGA-F-Stack")}
rng = np.random.default_rng(0)
for s in seeds:
    pr = {m: P(s, m).argmax(1) for m in cm_pool}
    for m, p in pr.items():
        cm_pool[m] += confusion_matrix(yte, p, labels=list(range(NC)))
        rec = recall_score(yte, p, average=None, labels=list(range(NC)), zero_division=0)
        pre = precision_score(yte, p, average=None, labels=list(range(NC)), zero_division=0)
        f1 = f1_score(yte, p, average=None, labels=list(range(NC)), zero_division=0)
        for c in range(NC):
            rows.append(dict(seed=s, model=m, cls=base.classes[c], support=int((yte == c).sum()), precision=pre[c] * 100,
                             recall=rec[c] * 100, f1=f1[c] * 100))
        mr.append(dict(seed=s, model=m, min_recall=rec.min() * 100, limiting_class=base.classes[rec.argmin()],
                       errors_in_limiting_class=int(((yte == rec.argmin()) & (p != yte)).sum()),
                       support=int((yte == rec.argmin()).sum())))
    raw.append(dict(seed=s, **{m: f1_score(yte, p, average="macro") * 100 for m, p in pr.items()}))
    a, b = pr["SBGA-F-Stack"] == yte, pr["CatBoost"] == yte
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    diffs = []
    for _ in range(1000):
        i = rng.integers(0, len(yte), len(yte))
        diffs.append(f1_score(yte[i], pr["SBGA-F-Stack"][i], average="macro") - f1_score(yte[i], pr["CatBoost"][i], average="macro"))
    lo, hi = np.percentile(diffs, [2.5, 97.5]) * 100
    mcn.append(dict(seed=s, stack_only_correct=n10, catboost_only_correct=n01,
                    mcnemar_p=binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0,
                    macro_F1_diff=raw[-1]["SBGA-F-Stack"] - raw[-1]["CatBoost"], boot_lo=lo, boot_hi=hi))
pc = pd.DataFrame(rows)
pc.to_csv(os.path.join(OUT, "audit_per_class_raw.csv"), index=False)
save("per_class", pc.groupby(["model", "cls"], sort=False).agg(support=("support", "first"), precision=("precision", "mean"),
                                                                 recall=("recall", "mean"), recall_sd=("recall", "std"), f1=("f1", "mean")).reset_index())
mrd = pd.DataFrame(mr)
mrd.to_csv(os.path.join(OUT, "audit_min_recall_raw.csv"), index=False)
rules = []
for m in cm_pool:
    cm = cm_pool[m]
    pooled = np.diag(cm) / cm.sum(1)
    mean_rec = pc[pc.model == m].groupby("cls", sort=False)["recall"].mean()
    k = int(pooled.argmin())
    n = cm.sum(1)[k]
    se = np.sqrt(pooled[k] * (1 - pooled[k]) / (n / len(seeds)))
    rules.append(dict(model=m, mean_of_seed_minima=mrd[mrd.model == m].min_recall.mean(), sd_of_seed_minima=mrd[mrd.model == m].min_recall.std(),
                      min_of_seed_averaged_class_recalls=mean_rec.min(), limiting_class_avg=mean_rec.idxmin(),
                      pooled_min_recall=pooled.min() * 100, limiting_class_pooled=base.classes[k],
                      limiting_class_test_support=int(n / len(seeds)), binomial_se_points=se * 100,
                      limiting_classes_by_seed=json.dumps(mrd[mrd.model == m].limiting_class.value_counts().to_dict())))
    np.save(os.path.join(OUT, f"cm_{m}.npy"), cm)
save("min_recall_rules", pd.DataFrame(rules))
save("raw_paired_scores", pd.DataFrame(raw))
save("mcnemar", pd.DataFrame(mcn))

# 7 ---- interleaved-stream stack
rows = []
perm_te = np.random.default_rng(13).permutation(len(pos["test"]))
perm_va = np.random.default_rng(12).permutation(len(pos["val"]))
for s in seeds:
    if not have(s, "SBGA-F_no_graph_interleaved"):
        continue
    cv, ct = P(s, "CatBoost", "val")[perm_va], P(s, "CatBoost")[perm_te]
    y_v, y_t = yva[perm_va], yte[perm_te]
    rows.append(dict(seed=s, model="CatBoost", macro_F1=f1_score(y_t, ct.argmax(1), average="macro") * 100))
    for m in ("SBGA-F", "SBGA-F_no_graph", "RNN", "BiRAGNN"):
        v, t = P(s, f"{m}_interleaved", "val"), P(s, f"{m}_interleaved")
        rows.append(dict(seed=s, model=m, macro_F1=f1_score(y_t, t.argmax(1), average="macro") * 100,
                         min_recall=recall_score(y_t, t.argmax(1), average=None).min() * 100))
        st_ = lr_stack([v, cv], [t, ct], y_v).argmax(1)
        rows.append(dict(seed=s, model=m + " + CatBoost (stacked)", macro_F1=f1_score(y_t, st_, average="macro") * 100,
                         min_recall=recall_score(y_t, st_, average=None).min() * 100))
if rows:
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "audit_interleaved_raw.csv"), index=False)
    save("interleaved", d.groupby("model", sort=False).agg(macro_F1=("macro_F1", "mean"), sd=("macro_F1", "std"),
                                                             min_recall=("min_recall", "mean"), n=("seed", "count")).reset_index())
print("audit done")
