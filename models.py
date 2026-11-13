"""
Model zoo for SOURCE_METHOD_REPRODUCTION: the paper's 6 benchmark IDSs (FNN, RNN,
CNN, Transformer, GNN, TGNN) plus the proposed BiRAGNN, built from shared, documented
building blocks (see SOURCE_PAPER_AUDIT.md "Partially specified": the paper gives no
closed-form equations for BiRAGNN, so this is our explicit, reproducible reconstruction
of its prose description).

Design (applies uniformly across all 7 models so differences are ONLY the ones the
paper claims matter):
  temporal encoder in {flat (FNN), lstm (RNN), cnn (CNN), transformer (Transformer/TGNN),
                        lstm_attn (BiRAGNN: recurrent + attention)}
  graph context     in {none (standalone/non-graph), uni (GNN/TGNN: in-neighbors only),
                        bi (BiRAGNN: in- AND out-neighbors, i.e. bidirectional flow)}
Neighbor context uses fixed TRAIN-only per-node feature prototypes (mean feature vector
of a node's training rows) aggregated through the static probabilistic graph adjacency,
matching the paper's node-embeddings-from-graph-structure description (Sec. III-A2a).
"""
import torch
import torch.nn as nn

ACTS = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}


def act_layer(name):
    return ACTS.get(name.lower(), nn.ReLU)()


class FlatEncoder(nn.Module):
    """FNN: uses only the current (last) timestep, no memory."""
    def __init__(self, in_dim, hidden, layers, dropout, activation):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        mods = []
        for i in range(layers):
            mods += [nn.Linear(dims[i], dims[i + 1]), act_layer(activation), nn.Dropout(dropout)]
        self.net = nn.Sequential(*mods)
        self.out_dim = hidden if layers > 0 else in_dim

    def forward(self, x_window):  # [B,W,F]
        x = x_window[:, -1, :]
        return self.net(x) if len(self.net) else x


class LSTMEncoder(nn.Module):
    """RNN benchmark: LSTM over the window, final hidden state, no attention."""
    def __init__(self, in_dim, hidden, layers, dropout, activation):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=max(1, layers), batch_first=True,
                             dropout=dropout if layers > 1 else 0.0)
        self.out_dim = hidden

    def forward(self, x_window):
        out, (h, c) = self.lstm(x_window)
        return h[-1]


class CNNEncoder(nn.Module):
    """CNN benchmark: stacked 1D conv over the time window, global-average-pooled."""
    def __init__(self, in_dim, hidden, layers, dropout, activation):
        super().__init__()
        mods = []
        c_in = in_dim
        for _ in range(max(1, layers)):
            mods += [nn.Conv1d(c_in, hidden, kernel_size=3, padding=1), act_layer(activation),
                     nn.Dropout(dropout)]
            c_in = hidden
        self.net = nn.Sequential(*mods)
        self.out_dim = hidden

    def forward(self, x_window):  # [B,W,F] -> conv wants [B,F,W]
        x = x_window.transpose(1, 2)
        x = self.net(x)
        return x.mean(dim=2)


class TransformerTemporalEncoder(nn.Module):
    """Transformer benchmark / TGNN's temporal part: self-attention encoder over the
    time window, feedforward (encoder-only), mean-pooled."""
    def __init__(self, in_dim, hidden, layers, dropout, activation):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=max(1, min(4, hidden // 8 or 1)), dim_feedforward=hidden * 2,
            dropout=dropout, activation="relu" if activation.lower() == "relu" else "gelu",
            batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, layers))
        self.out_dim = hidden

    def forward(self, x_window):
        x = self.proj(x_window)
        x = self.encoder(x)
        return x.mean(dim=1)


class LSTMAttnEncoder(nn.Module):
    """BiRAGNN's temporal part: LSTM (recurrence) + additive attention pooling over
    timesteps ("assigning higher weights to impactful instances", Sec III-A2e)."""
    def __init__(self, in_dim, hidden, layers, dropout, activation):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=max(1, layers), batch_first=True,
                             dropout=dropout if layers > 1 else 0.0)
        self.attn = nn.Sequential(nn.Linear(hidden, hidden), act_layer(activation), nn.Linear(hidden, 1))
        self.out_dim = hidden

    def forward(self, x_window):
        out, _ = self.lstm(x_window)              # [B,W,H]
        scores = self.attn(out).squeeze(-1)        # [B,W]
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B,W,1]
        context = (out * weights).sum(dim=1)        # [B,H]
        return context


TEMPORAL_ENCODERS = {
    "flat": FlatEncoder, "lstm": LSTMEncoder, "cnn": CNNEncoder,
    "transformer": TransformerTemporalEncoder, "lstm_attn": LSTMAttnEncoder,
}


class IDSModel(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout, activation,
                 temporal="flat", graph="none", n_classes=1):
        super().__init__()
        self.temporal_encoder = TEMPORAL_ENCODERS[temporal](in_dim, hidden, layers, dropout, activation)
        self.graph = graph
        combine_dim = self.temporal_encoder.out_dim
        if graph in ("uni", "bi"):
            self.neigh_in_proj = nn.Linear(in_dim, hidden)
            combine_dim += hidden
        if graph == "bi":
            self.neigh_out_proj = nn.Linear(in_dim, hidden)
            combine_dim += hidden
        self.dense = nn.Sequential(nn.Linear(combine_dim, hidden), act_layer(activation), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x_window, neigh_in=None, neigh_out=None):
        h = self.temporal_encoder(x_window)
        parts = [h]
        if self.graph in ("uni", "bi"):
            parts.append(self.neigh_in_proj(neigh_in))
        if self.graph == "bi":
            parts.append(self.neigh_out_proj(neigh_out))
        combined = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        z = self.dense(combined)
        return self.out(z)


# name -> (temporal, graph) matching the paper's architectural claims
MODEL_SPECS = {
    "FNN":      dict(temporal="flat",        graph="none"),
    "RNN":      dict(temporal="lstm",        graph="none"),
    "CNN":      dict(temporal="cnn",         graph="none"),
    "Transformer": dict(temporal="transformer", graph="none"),
    "GNN":      dict(temporal="flat",        graph="uni"),
    "TGNN":     dict(temporal="transformer", graph="uni"),
    "BiRAGNN":  dict(temporal="lstm_attn",   graph="bi"),
}
GRAPH_MODELS = {"GNN", "TGNN", "BiRAGNN"}
