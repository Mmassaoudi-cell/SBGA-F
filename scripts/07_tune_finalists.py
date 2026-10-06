"""
Stage 3: light Optuna (TPE) tuning of the 2 candidates advancing past screening as
serious finalists -- B_SBGAF (best novelty/minority-recall/topology-transfer profile)
and A_GTCascade (best efficiency profile). C_TCNGXD is documented in
MODEL_SELECTION_REPORT.md as screened-but-not-tuned (narrowest novelty contribution
of the three, and its marginal macro-F1 edge over B is within seed-to-seed noise).

Validation-only objective (macro-F1). Modest budget (20 trials each), TPE sampler.
"""
import os
import sys
import time

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
import importlib
scr = importlib.import_module("04_candidate_screening")
from candidate_models import CandidateB_SBGAF, CandidateA_GraphBranch, FocalLoss, class_weights_sqrt

DEVICE = scr.DEVICE
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def to_t(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype)


def objective_B(trial, data):
    hidden = trial.suggest_categorical("hidden", [64, 128, 192])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    gamma = trial.suggest_float("gamma", 1.0, 3.0)
    lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    k = scr.K_TOPK  # keep k fixed (already precomputed in data); tune the rest

    torch.manual_seed(0); np.random.seed(0)
    tr, va = data["splits"]["train"], data["splits"]["val"]
    model = CandidateB_SBGAF(data["in_dim"], data["struct_dim"], hidden, 2, dropout, "relu",
                              data["n_classes"], k=k).to(DEVICE)
    cw = class_weights_sqrt(tr["y"], data["n_classes"])
    loss_fn = FocalLoss(cw, gamma=gamma).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xtr = to_t(tr["X_win"]); Str = to_t(tr["struct_feat"])
    Nin_tr = to_t(tr["topk_in_feats"]); Ninw_tr = to_t(tr["topk_in_w"])
    Nout_tr = to_t(tr["topk_out_feats"]); Noutw_tr = to_t(tr["topk_out_w"])
    ytr = to_t(tr["y"], torch.long)
    Xva = to_t(va["X_win"]).to(DEVICE); Sva = to_t(va["struct_feat"]).to(DEVICE)
    Nin_va = to_t(va["topk_in_feats"]).to(DEVICE); Ninw_va = to_t(va["topk_in_w"]).to(DEVICE)
    Nout_va = to_t(va["topk_out_feats"]).to(DEVICE); Noutw_va = to_t(va["topk_out_w"]).to(DEVICE)
    yva = va["y"]

    n = len(Xtr); best_f1, bad = -1, 0
    for epoch in range(12):
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
        vf1 = f1_score(yva, vpred, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
        else:
            bad += 1
            if bad >= 4:
                break
        trial.report(vf1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best_f1


def objective_A(trial, data):
    hidden = trial.suggest_categorical("hidden", [32, 64, 96])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    n_est = trial.suggest_categorical("gbm_n_estimators", [100, 200, 300])

    torch.manual_seed(0); np.random.seed(0)
    tr, va = data["splits"]["train"], data["splits"]["val"]
    import lightgbm as lgb
    gbm = lgb.LGBMClassifier(n_estimators=n_est, num_leaves=63, learning_rate=0.05,
                              objective="multiclass", num_class=data["n_classes"],
                              class_weight="balanced", random_state=0, verbosity=-1)
    gbm.fit(tr["X_flat"], tr["y"])

    model = CandidateA_GraphBranch(data["in_dim"], hidden, 2, dropout, "relu", data["n_classes"]).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr = to_t(tr["X_win"]); Nin_tr = to_t(tr["neigh_in"]); Nout_tr = to_t(tr["neigh_out"]); ytr = to_t(tr["y"], torch.long)
    Xva = to_t(va["X_win"]).to(DEVICE); Nin_va = to_t(va["neigh_in"]).to(DEVICE); Nout_va = to_t(va["neigh_out"]).to(DEVICE)
    yva = va["y"]
    n = len(Xtr); best_f1, bad = -1, 0
    for epoch in range(10):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            logits = model(Xtr[idx].to(DEVICE), Nin_tr[idx].to(DEVICE), Nout_tr[idx].to(DEVICE))
            loss = loss_fn(logits, ytr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            graph_pred = model(Xva, Nin_va, Nout_va).argmax(1).cpu().numpy()
        vf1 = f1_score(yva, graph_pred, average="macro", zero_division=0)
        best_f1 = max(best_f1, vf1)

    gbm_proba = gbm.predict_proba(va["X_flat"])
    gbm_conf = gbm_proba.max(1); gbm_pred = gbm_proba.argmax(1)
    with torch.no_grad():
        graph_proba = F.softmax(model(Xva, Nin_va, Nout_va), dim=1).cpu().numpy()
    graph_pred = graph_proba.argmax(1)
    best_tau_f1 = -1
    for tau in np.linspace(0.3, 0.95, 14):
        routed = gbm_conf < tau
        pred = np.where(routed, graph_pred, gbm_pred)
        f1 = f1_score(yva, pred, average="macro", zero_division=0)
        best_tau_f1 = max(best_tau_f1, f1)
    return best_tau_f1


def main():
    data = scr.prepare_data()
    print("Tuning B_SBGAF (20 trials)...", flush=True)
    study_b = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study_b.optimize(lambda t: objective_B(t, data), n_trials=20, show_progress_bar=False)
    print("Best B params:", study_b.best_params, "val macroF1:", study_b.best_value, flush=True)

    print("Tuning A_GTCascade (15 trials)...", flush=True)
    study_a = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study_a.optimize(lambda t: objective_A(t, data), n_trials=15, show_progress_bar=False)
    print("Best A params:", study_a.best_params, "val macroF1:", study_a.best_value, flush=True)

    summary = pd.DataFrame([
        dict(candidate="B_SBGAF", best_val_macroF1=study_b.best_value, **study_b.best_params),
        dict(candidate="A_GTCascade", best_val_macroF1=study_a.best_value, **study_a.best_params),
    ])
    summary.to_csv(os.path.join(RESULTS_DIR, "finalist_tuning_summary.csv"), index=False)

    trials_b = study_b.trials_dataframe()
    trials_b.to_csv(os.path.join(RESULTS_DIR, "tuning_trials_B.csv"), index=False)
    trials_a = study_a.trials_dataframe()
    trials_a.to_csv(os.path.join(RESULTS_DIR, "tuning_trials_A.csv"), index=False)
    print("DONE. Saved results/finalist_tuning_summary.csv")


if __name__ == "__main__":
    main()
