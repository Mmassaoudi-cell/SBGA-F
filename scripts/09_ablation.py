"""
Ablation study (Section 19): Full SBGA-F model vs. each component removed/swapped,
on the untouched TEST set, 3 seeds each. Isolates whether each claimed mechanism
(bidirectional attention, structural/topology-invariant node features, LEARNED sparse
top-k attention vs. plain fixed-weight aggregation, LSTM+attention vs. TCN temporal
encoder) is actually earning its cost -- exactly the kind of evidence the source
paper's own BiRAGNN never provides (W4 in SOURCE_WEAKNESS_ANALYSIS.md).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
import importlib
scr = importlib.import_module("04_candidate_screening")
from candidate_models import CandidateB_SBGAF, FocalLoss, class_weights_sqrt

DEVICE = scr.DEVICE
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = [42, 123, 2024]

ARMS = {
    "Full_SBGA-F":            dict(use_bidirectional=True,  use_structural=True,  use_sparse_attn=True,  temporal_type="lstm_attn", use_focal=True),
    "minus_bidirectional":    dict(use_bidirectional=False, use_structural=True,  use_sparse_attn=True,  temporal_type="lstm_attn", use_focal=True),
    "minus_structural_feats": dict(use_bidirectional=True,  use_structural=False, use_sparse_attn=True,  temporal_type="lstm_attn", use_focal=True),
    "dense_instead_of_sparse_attn": dict(use_bidirectional=True, use_structural=True, use_sparse_attn=False, temporal_type="lstm_attn", use_focal=True),
    "TCN_instead_of_LSTMAttn":dict(use_bidirectional=True,  use_structural=True,  use_sparse_attn=True,  temporal_type="tcn",       use_focal=True),
    "CE_instead_of_focal":    dict(use_bidirectional=True,  use_structural=True,  use_sparse_attn=True,  temporal_type="lstm_attn", use_focal=False),
}


def to_t(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype)


def run_arm(data, seed, arm_cfg):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = data["splits"]["train"], data["splits"]["val"], data["splits"]["test"]
    model = CandidateB_SBGAF(data["in_dim"], data["struct_dim"], 192, 2, 0.3397475692650171, "relu",
                              data["n_classes"], k=scr.K_TOPK,
                              use_bidirectional=arm_cfg["use_bidirectional"],
                              use_structural=arm_cfg["use_structural"],
                              use_sparse_attn=arm_cfg["use_sparse_attn"],
                              temporal_type=arm_cfg["temporal_type"]).to(DEVICE)
    cw = class_weights_sqrt(tr["y"], data["n_classes"])
    if arm_cfg["use_focal"]:
        loss_fn = FocalLoss(cw, gamma=1.9229587245058637).to(DEVICE)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(weight=to_t(cw).to(DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=0.0018098827092128786)

    def bt(s):
        return (to_t(s["X_win"]), to_t(s["struct_feat"]), to_t(s["topk_in_feats"]), to_t(s["topk_in_w"]),
                to_t(s["topk_out_feats"]), to_t(s["topk_out_w"]), to_t(s["y"], torch.long))

    Xtr, Str, Nin_tr, Ninw_tr, Nout_tr, Noutw_tr, ytr = bt(tr)
    Xva, Sva, Nin_va, Ninw_va, Nout_va, Noutw_va, yva = [t.to(DEVICE) for t in bt(va)]
    Xte, Ste, Nin_te, Ninw_te, Nout_te, Noutw_te, yte = [t.to(DEVICE) for t in bt(te)]

    n = len(Xtr); best_f1, best_state, bad = -1, None, 0
    t0 = time.time()
    for epoch in range(25):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            logits = model(Xtr[idx].to(DEVICE), Str[idx].to(DEVICE), Nin_tr[idx].to(DEVICE),
                            Ninw_tr[idx].to(DEVICE), Nout_tr[idx].to(DEVICE), Noutw_tr[idx].to(DEVICE))
            loss = loss_fn(logits, ytr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vpred = model(Xva, Sva, Nin_va, Ninw_va, Nout_va, Noutw_va).argmax(1).cpu().numpy()
        vf1 = f1_score(yva.cpu().numpy(), vpred, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, best_state, bad = vf1, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 7:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        tproba = F.softmax(model(Xte, Ste, Nin_te, Ninw_te, Nout_te, Noutw_te), dim=1).cpu().numpy()
    tpred = tproba.argmax(1)
    n_params = sum(p.numel() for p in model.parameters())
    m = scr.eval_metrics(yte.cpu().numpy(), tpred, tproba, data["n_classes"])
    m.update(train_time_sec=train_time, n_params=n_params)
    return m


def main():
    data = scr.prepare_data()
    results = []
    for arm_name, cfg in ARMS.items():
        for seed in SEEDS:
            print(f"-- {arm_name} seed={seed}", flush=True)
            m = run_arm(data, seed, cfg)
            m["arm"] = arm_name
            m["seed"] = seed
            print(f"   macroF1={m['macro_F1']:.2f} minRecall={m['min_class_recall']:.2f} params={m['n_params']}", flush=True)
            results.append(m)
            pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"), index=False)
    print("DONE. Saved results/ablation_results.csv")


if __name__ == "__main__":
    main()
