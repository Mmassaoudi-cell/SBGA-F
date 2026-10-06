"""
Topology-transfer ablation (addresses Methodology reviewer's Weakness #3): trains
SBGA-F variants WITHOUT structural node embeddings and WITHOUT bidirectionality, then
runs each through the SAME zero-shot cross-topology transfer test as the full model
(train on 100-node graph, evaluate with NO retraining on 8-node/50-node graphs). This
directly tests whether structural embeddings are the mechanism behind transfer, or
whether transfer would happen anyway (e.g. from more training data alone).

If "minus structural features" transfers nearly as well as the full model, the
paper's causal claim (structural embeddings ENABLE transfer) is not supported by this
data and must be revised. If it transfers markedly worse, the claim is validated.
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
rob = importlib.import_module("10_robustness_generalization")
from candidate_models import CandidateB_SBGAF, FocalLoss, class_weights_sqrt

DEVICE = scr.DEVICE
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 42

ARMS = {
    "Full_SBGA-F":            dict(use_bidirectional=True,  use_structural=True,  use_sparse_attn=True),
    "minus_structural_feats": dict(use_bidirectional=True,  use_structural=False, use_sparse_attn=True),
    "minus_bidirectional":    dict(use_bidirectional=False, use_structural=True,  use_sparse_attn=True),
}


def to_t(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype)


def train_arm(data, arm_cfg, seed=SEED, max_epochs=30, patience=8):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = data["splits"]["train"], data["splits"]["val"]
    model = CandidateB_SBGAF(data["in_dim"], data["struct_dim"], 192, 2, 0.3397475692650171, "relu",
                              data["n_classes"], k=scr.K_TOPK,
                              use_bidirectional=arm_cfg["use_bidirectional"],
                              use_structural=arm_cfg["use_structural"],
                              use_sparse_attn=arm_cfg["use_sparse_attn"]).to(DEVICE)
    cw = class_weights_sqrt(tr["y"], data["n_classes"])
    loss_fn = FocalLoss(cw, gamma=1.9229587245058637).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.0018098827092128786)

    def bt(s):
        return (to_t(s["X_win"]), to_t(s["struct_feat"]), to_t(s["topk_in_feats"]), to_t(s["topk_in_w"]),
                to_t(s["topk_out_feats"]), to_t(s["topk_out_w"]), to_t(s["y"], torch.long))

    Xtr, Str, Nin_tr, Ninw_tr, Nout_tr, Noutw_tr, ytr = bt(tr)
    Xva, Sva, Nin_va, Ninw_va, Nout_va, Noutw_va, yva = [t.to(DEVICE) for t in bt(va)]

    n = len(Xtr); best_f1, best_state, bad = -1, None, 0
    for epoch in range(max_epochs):
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
            if bad >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    return model


def eval_zero_shot(model, target_data):
    s = target_data["splits"]["test"]
    with torch.no_grad():
        proba = F.softmax(model(to_t(s["X_win"]).to(DEVICE), to_t(s["struct_feat"]).to(DEVICE),
                                 to_t(s["topk_in_feats"]).to(DEVICE), to_t(s["topk_in_w"]).to(DEVICE),
                                 to_t(s["topk_out_feats"]).to(DEVICE), to_t(s["topk_out_w"]).to(DEVICE)), dim=1).cpu().numpy()
    pred = proba.argmax(1)
    return scr.eval_metrics(s["y"], pred, proba, target_data["n_classes"])


def main():
    print("preparing 100-node data...", flush=True)
    data_100 = scr.prepare_data()
    print("preparing 8-node and 50-node target data...", flush=True)
    data_8 = rob.prepare_system(8)
    data_50 = rob.prepare_system(50)

    results = []
    for arm_name, cfg in ARMS.items():
        print(f"\n== training arm: {arm_name} ==", flush=True)
        model = train_arm(data_100, cfg)

        m_same = eval_zero_shot(model, data_100)
        m_same.update(arm=arm_name, target_system=100, mode="same_topology")
        results.append(m_same)
        print(f"  same-topology (100-node): macroF1={m_same['macro_F1']:.2f}", flush=True)

        for target_size, target_data in [(8, data_8), (50, data_50)]:
            m_zs = eval_zero_shot(model, target_data)
            m_zs.update(arm=arm_name, target_system=target_size, mode="zero_shot_transfer")
            results.append(m_zs)
            print(f"  zero-shot -> {target_size}-node: macroF1={m_zs['macro_F1']:.2f}", flush=True)

        pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "transfer_ablation_results.csv"), index=False)

    print("\nDONE. Saved results/transfer_ablation_results.csv")


if __name__ == "__main__":
    main()
