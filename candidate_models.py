"""
Implementations of the 4 hybrid candidates from MODEL_CANDIDATES.md. Each is a
genuinely different mechanism from the SOURCE_METHOD_REPRODUCTION models (models.py),
not "BiRAGNN plus a layer":

  Candidate A (GT-Cascade):   LightGBM confidence gate -> compact dense BiGAT for hard cases
  Candidate B (SBGA-F):       LEARNED top-k sparse bidirectional graph attention (not the
                               static-probability-weighted aggregation used in models.py) +
                               structural/positional node features + multiclass focal loss
  Candidate C (TCN-GXD):      dilated causal TCN (parallel, no recurrence) + pooled graph
                               cross-attention + distillation from a GBM teacher
  Candidate D (EBGN):         bidirectional dense graph attention + evidential (Dirichlet) head
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import act_layer, LSTMAttnEncoder


# ----------------------------------------------------------------------------------
# Shared: multiclass focal loss with sqrt-frequency class weights
# ----------------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, class_weights, gamma=2.0):
        super().__init__()
        self.register_buffer("w", torch.as_tensor(class_weights, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        w_t = self.w[target]
        loss = -w_t * ((1 - p_t) ** self.gamma) * logp_t
        return loss.mean()


def class_weights_sqrt(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    return w.astype(np.float32)


# ----------------------------------------------------------------------------------
# Candidate B: Sparse Bidirectional Graph Attention + Focal Loss (SBGA-F)
# ----------------------------------------------------------------------------------
class SparseTopKAttention(nn.Module):
    """Learned attention over each node's precomputed top-k neighbors (indices/weights
    fixed from the static graph; attention SCORES are learned, unlike models.py's plain
    fixed-weight aggregation)."""
    def __init__(self, feat_dim, hidden):
        super().__init__()
        self.wq = nn.Linear(hidden, hidden)
        self.wk = nn.Linear(feat_dim, hidden)
        self.wv = nn.Linear(feat_dim, hidden)
        self.edge_bias = nn.Linear(1, 1)
        self.out_dim = hidden

    def forward(self, query, neighbor_feats, edge_w):
        # query: [B,H], neighbor_feats: [B,k,F], edge_w: [B,k]
        q = self.wq(query).unsqueeze(1)               # [B,1,H]
        k = self.wk(neighbor_feats)                    # [B,k,H]
        v = self.wv(neighbor_feats)                     # [B,k,H]
        scores = (q * k).sum(-1) / (k.shape[-1] ** 0.5)  # [B,k]
        scores = scores + self.edge_bias(edge_w.unsqueeze(-1)).squeeze(-1)
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B,k,1]
        return (attn * v).sum(dim=1)  # [B,H]


class DenseNeighborAgg(nn.Module):
    """Ablation arm: plain fixed-weight (non-learned) dense aggregation over ALL
    neighbors via the static graph probabilities -- i.e. models.py's mechanism,
    substituted in place of SparseTopKAttention to isolate whether LEARNED, SPARSE
    attention is actually earning its cost over the source paper's plain aggregation."""
    def __init__(self, feat_dim, hidden):
        super().__init__()
        self.proj = nn.Linear(feat_dim, hidden)
        self.out_dim = hidden

    def forward(self, query, neighbor_feats, edge_w):
        # neighbor_feats: [B,k,F] (here we just reuse the same top-k bank but weight
        # by the FIXED edge_w instead of a learned attention score, then sum)
        w = edge_w / (edge_w.sum(dim=1, keepdim=True) + 1e-8)
        agg = (neighbor_feats * w.unsqueeze(-1)).sum(dim=1)
        return self.proj(agg)


class CandidateB_SBGAF(nn.Module):
    def __init__(self, in_dim, struct_dim, hidden, layers, dropout, activation, n_classes, k=6,
                 use_bidirectional=True, use_structural=True, use_sparse_attn=True, temporal_type="lstm_attn"):
        super().__init__()
        self.use_bidirectional = use_bidirectional
        self.use_structural = use_structural
        self.temporal_type = temporal_type

        if temporal_type == "tcn":
            self.temporal = TCNEncoder(in_dim, hidden, layers, dropout, activation)  # defined later in this module; resolved at call time
        else:
            self.temporal = LSTMAttnEncoder(in_dim, hidden, layers, dropout, activation)

        if use_structural:
            self.struct_proj = nn.Linear(struct_dim, hidden)

        attn_cls = SparseTopKAttention if use_sparse_attn else DenseNeighborAgg
        self.attn_in = attn_cls(in_dim, hidden)
        if use_bidirectional:
            self.attn_out = attn_cls(in_dim, hidden)

        combine_dim = self.temporal.out_dim + hidden + (hidden if use_bidirectional else 0) + (hidden if use_structural else 0)
        self.dense = nn.Sequential(nn.Linear(combine_dim, hidden), act_layer(activation), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x_window, struct_feat, neigh_in_feats, neigh_in_w, neigh_out_feats, neigh_out_w):
        h_t = self.temporal(x_window)
        parts = [h_t]
        if self.use_structural:
            parts.append(self.struct_proj(struct_feat))
        parts.append(self.attn_in(h_t, neigh_in_feats, neigh_in_w))
        if self.use_bidirectional:
            parts.append(self.attn_out(h_t, neigh_out_feats, neigh_out_w))
        combined = torch.cat(parts, dim=1)
        z = self.dense(combined)
        return self.out(z)


# ----------------------------------------------------------------------------------
# Candidate A: GBM-Gated Two-Path Cascade (GT-Cascade). The graph branch reuses the
# dense bidirectional aggregation style from models.py (smaller hidden dim) since its
# whole point is to be a *cheap* fallback, not a novel spatial mechanism -- the
# innovation here is the confidence-gated routing itself, not the branch architecture.
# ----------------------------------------------------------------------------------
class CandidateA_GraphBranch(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout, activation, n_classes):
        super().__init__()
        self.temporal = LSTMAttnEncoder(in_dim, hidden, layers, dropout, activation)
        self.neigh_in_proj = nn.Linear(in_dim, hidden)
        self.neigh_out_proj = nn.Linear(in_dim, hidden)
        combine_dim = self.temporal.out_dim + hidden + hidden
        self.dense = nn.Sequential(nn.Linear(combine_dim, hidden), act_layer(activation), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x_window, neigh_in, neigh_out):
        h_t = self.temporal(x_window)
        h_in = self.neigh_in_proj(neigh_in)
        h_out = self.neigh_out_proj(neigh_out)
        z = self.dense(torch.cat([h_t, h_in, h_out], dim=1))
        return self.out(z)


# ----------------------------------------------------------------------------------
# Candidate C: TCN + pooled graph cross-attention + distillation (TCN-GXD)
# ----------------------------------------------------------------------------------
class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=pad, dilation=dilation)
        self.pad = pad
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):  # [B,C,T]
        out = self.conv(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]  # causal: trim right padding
        out = self.drop(self.act(out))
        return out + self.res(x)


class TCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout, activation):
        super().__init__()
        blocks = []
        c_in = in_dim
        for l in range(max(1, layers)):
            blocks.append(TCNBlock(c_in, hidden, kernel_size=3, dilation=2 ** l, dropout=dropout))
            c_in = hidden
        self.net = nn.Sequential(*blocks)
        self.out_dim = hidden

    def forward(self, x_window):  # [B,W,F] -> [B,F,W]
        x = x_window.transpose(1, 2)
        x = self.net(x)
        return x[:, :, -1]  # last (causal, most recent) timestep


class CrossAttnPool(nn.Module):
    def __init__(self, feat_dim, hidden):
        super().__init__()
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(feat_dim, hidden)
        self.v = nn.Linear(feat_dim, hidden)

    def forward(self, query, proto_all, node_w):
        # proto_all: [n_nodes, F] shared across batch; node_w: [B, n_nodes] weights (adjacency row)
        q = self.q(query)                       # [B,H]
        k = self.k(proto_all)                    # [n_nodes,H]
        v = self.v(proto_all)                     # [n_nodes,H]
        scores = q @ k.T                          # [B,n_nodes]
        scores = scores.masked_fill(node_w <= 0, float("-inf"))
        attn = torch.softmax(scores, dim=1)
        attn = torch.nan_to_num(attn, nan=0.0)
        return attn @ v                            # [B,H]


class CandidateC_TCNGXD(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout, activation, n_classes):
        super().__init__()
        self.temporal = TCNEncoder(in_dim, hidden, layers, dropout, activation)
        self.cross_attn = CrossAttnPool(in_dim, hidden)
        combine_dim = self.temporal.out_dim + hidden
        self.dense = nn.Sequential(nn.Linear(combine_dim, hidden), act_layer(activation), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x_window, proto_all, node_w):
        h_t = self.temporal(x_window)
        h_g = self.cross_attn(h_t, proto_all, node_w)
        z = self.dense(torch.cat([h_t, h_g], dim=1))
        return self.out(z)


def distillation_loss(student_logits, teacher_probs, target, hard_weight=0.5, T=2.0):
    ce = F.cross_entropy(student_logits, target)
    student_logp = F.log_softmax(student_logits / T, dim=1)
    teacher_p = torch.clamp(teacher_probs, min=1e-8)
    kd = F.kl_div(student_logp, teacher_p, reduction="batchmean") * (T * T)
    return hard_weight * ce + (1 - hard_weight) * kd


# ----------------------------------------------------------------------------------
# Candidate D: Evidential Bidirectional Graph Network (EBGN) -- smoke-test only
# ----------------------------------------------------------------------------------
class CandidateD_EBGN(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout, activation, n_classes):
        super().__init__()
        self.temporal = LSTMAttnEncoder(in_dim, hidden, layers, dropout, activation)
        self.neigh_in_proj = nn.Linear(in_dim, hidden)
        self.neigh_out_proj = nn.Linear(in_dim, hidden)
        combine_dim = self.temporal.out_dim + hidden + hidden
        self.dense = nn.Sequential(nn.Linear(combine_dim, hidden), act_layer(activation), nn.Dropout(dropout))
        self.evidence = nn.Linear(hidden, n_classes)  # Dirichlet evidence head

    def forward(self, x_window, neigh_in, neigh_out):
        h_t = self.temporal(x_window)
        h_in = self.neigh_in_proj(neigh_in)
        h_out = self.neigh_out_proj(neigh_out)
        z = self.dense(torch.cat([h_t, h_in, h_out], dim=1))
        evidence = F.softplus(self.evidence(z))
        return evidence  # alpha = evidence + 1


def evidential_loss(evidence, target, n_classes, epoch, annealing_epochs=10):
    alpha = evidence + 1
    S = alpha.sum(dim=1, keepdim=True)
    y = F.one_hot(target, n_classes).float()
    err = ((y - alpha / S) ** 2).sum(dim=1)
    var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(dim=1)
    loss = (err + var).mean()
    # KL regularizer toward uniform Dirichlet, annealed in
    alpha_tilde = y + (1 - y) * alpha
    kl = _dirichlet_kl(alpha_tilde, n_classes)
    lam = min(1.0, epoch / annealing_epochs)
    return loss + lam * kl.mean()


def _dirichlet_kl(alpha, n_classes):
    beta = torch.ones_like(alpha)
    S_alpha = alpha.sum(dim=1, keepdim=True)
    S_beta = beta.sum(dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    lnB_uni = torch.lgamma(beta).sum(dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = ((alpha - beta) * (dg1 - dg0)).sum(dim=1, keepdim=True) + lnB + lnB_uni
    return kl.squeeze(1)
