"""Round-2 experiment G: memory footprint and CPU-only cost of the deployable configurations.
Reports parameter counts, serialized model size, graph-context size, peak GPU memory during inference,
resident memory of a CPU-only process, and CPU-only latency (1 and 8 threads).
Run on an idle machine. Usage: python 38_r2_memory.py"""
import io, os, sys, time, tempfile
import numpy as np
import pandas as pd
import psutil
import torch
sys.path.insert(0, os.path.dirname(__file__))
import exp_common as E
from models import IDSModel, MODEL_SPECS

OUT = os.path.join(E.RESULTS_DIR, "rev2")
os.makedirs(OUT, exist_ok=True)
rows = []
proc = psutil.Process()
rss0 = proc.memory_info().rss / 2**20
base = E.load_base()
data = E.build_dataset(base, 100)
tr, te, ctx = data["splits"]["train"], data["splits"]["test"], data["ctx"]
F_, NC = base.X.shape[1], base.n_classes


def state_mb(m):
    b = io.BytesIO()
    torch.save(m.state_dict(), b)
    return b.getbuffer().nbytes / 2**20


def build(name):
    if name == "SBGA-F":
        return E.Predictor("sbgaf", E.SBGAFv2(F_, 4, NC, E.HP["hidden"], E.HP["layers"], E.HP["dropout"]), base)
    spec = MODEL_SPECS[name]
    return E.Predictor(f"ids:{name}", IDSModel(F_, 128, 3, 0.2, "relu", temporal=spec["temporal"], graph=spec["graph"], n_classes=NC), base)


ctx_mb = sum(t.numel() * t.element_size() for t in (ctx.proto, ctx.struct, ctx.in_idx, ctx.in_w, ctx.out_idx, ctx.out_w)) / 2**20
rows.append(dict(item="Graph context (100 nodes: prototypes, descriptors, top-k lists)", value=ctx_mb, unit="MB"))
X_gpu = torch.as_tensor(base.X).to(E.DEVICE)
for name in ("SBGA-F", "BiRAGNN", "RNN"):
    pred = build(name)
    n_par = sum(p.numel() for p in pred.model.parameters())
    rows.append(dict(item=f"{name} parameters", value=n_par, unit="count"))
    rows.append(dict(item=f"{name} serialized weights", value=state_mb(pred.model), unit="MB"))
    pred.model.to(E.DEVICE).eval()
    for bs in (1, 1024, 4096):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        m0 = torch.cuda.memory_allocated()
        with torch.no_grad():
            idx = torch.arange(0, bs, device=E.DEVICE)
            pred.logits(te, ctx, X_gpu, idx)
        rows.append(dict(item=f"{name} peak GPU memory above resident tensors, batch {bs}",
                         value=(torch.cuda.max_memory_allocated() - m0) / 2**20, unit="MB"))
    rows.append(dict(item=f"{name} resident GPU memory (weights)", value=sum(p.numel() * 4 for p in pred.model.parameters()) / 2**20, unit="MB"))

# tree models
Xtr, Xte = base.X[tr.pos], base.X[te.pos]
cb = E.fit_classical("CatBoost", 42, base, Xtr, tr.y_np)
lg = E.fit_classical("LightGBM", 42, base, Xtr, tr.y_np)
with tempfile.TemporaryDirectory() as td:
    cb.save_model(os.path.join(td, "cb.cbm"))
    rows.append(dict(item="CatBoost serialized model", value=os.path.getsize(os.path.join(td, "cb.cbm")) / 2**20, unit="MB"))
    lg.booster_.save_model(os.path.join(td, "lg.txt"))
    rows.append(dict(item="LightGBM serialized model", value=os.path.getsize(os.path.join(td, "lg.txt")) / 2**20, unit="MB"))

# CPU-only cost
cpu = torch.device("cpu")
X_cpu = torch.as_tensor(base.X)
for name in ("SBGA-F", "BiRAGNN"):
    pred = build(name)
    pred.model.eval()

    class C:
        pass
    c = C()
    for k in ("struct", "proto", "in_idx", "in_w", "in_m", "out_idx", "out_w", "out_m", "neigh_in", "neigh_out"):
        setattr(c, k, getattr(ctx, k).cpu())
    s = E.Split()
    s.node, s.win = te.node.cpu(), te.win.cpu()
    for threads in (1, 8):
        torch.set_num_threads(threads)
        for bs, reps in ((1, 300), (256, 40)):
            with torch.no_grad():
                for _ in range(5):
                    pred.logits(s, c, X_cpu, torch.arange(0, bs))
                t0 = time.perf_counter()
                for r in range(reps):
                    pred.logits(s, c, X_cpu, torch.arange(r * bs, (r + 1) * bs))
                dt = (time.perf_counter() - t0) / (reps * bs) * 1000
            rows.append(dict(item=f"{name} CPU latency, {threads} thread(s), batch {bs}", value=dt, unit="ms/event"))
for threads in (1, 8):
    t0 = time.perf_counter()
    for i in range(300):
        cb.predict_proba(Xte[i:i + 1], thread_count=threads)
    rows.append(dict(item=f"CatBoost CPU latency, {threads} thread(s), batch 1", value=(time.perf_counter() - t0) / 300 * 1000, unit="ms/event"))
    t0 = time.perf_counter()
    cb.predict_proba(Xte, thread_count=threads)
    rows.append(dict(item=f"CatBoost CPU latency, {threads} thread(s), full test partition", value=(time.perf_counter() - t0) / len(Xte) * 1000, unit="ms/event"))
rows.append(dict(item="Process resident memory before loading data", value=rss0, unit="MB"))
rows.append(dict(item="Process resident memory with data, models and graph context loaded", value=proc.memory_info().rss / 2**20, unit="MB"))
rows.append(dict(item="Feature matrix held in memory (297,394 x 54 float32)", value=base.X.nbytes / 2**20, unit="MB"))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "memory.csv"), index=False)
print(df.to_string(index=False))
