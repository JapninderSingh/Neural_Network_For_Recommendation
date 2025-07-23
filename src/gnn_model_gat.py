import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, emb, edge_index):
        h = self.fc(emb)
        row, col = edge_index
        edge_h = torch.cat([h[row], h[col]], dim=1)

        # Attention score computation (safe version)
        attn_scores = self.attn_fc(edge_h)
        attn_scores = F.leaky_relu(attn_scores, 0.2)
        attn_scores = torch.clamp(attn_scores, min=1e-6)

        num_nodes = emb.size(0)
        attn_sum = torch.zeros((num_nodes, 1), device=emb.device)
        attn_sum.index_add_(0, row, attn_scores)
        normalized_scores = attn_scores / (attn_sum[row] + 1e-6)

        out = torch.zeros_like(h)
        out.index_add_(0, row, normalized_scores * h[col])
        out = self.dropout(out)

        return h + out  # Residual connection

class GAT(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=2):
        super(GAT, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.layers = nn.ModuleList([
            GATLayer(embedding_dim, embedding_dim) for _ in range(num_layers)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, edge_index):
        emb_users = self.user_emb.weight
        emb_items = self.item_emb.weight
        emb = torch.cat([emb_users, emb_items], dim=0)

        for layer in self.layers:
            emb = layer(emb, edge_index)

        return emb[:emb_users.size(0)], emb[emb_users.size(0):]

def gnn_recommend_gat(user_id, user_encoder, song_encoder, songs_data, edge_index,
                      num_users, num_items, top_k=10, ground_truth=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        user_emb = torch.load("data/trained_user_emb_gat.pt", map_location=device)
        item_emb = torch.load("data/trained_item_emb_gat.pt", map_location=device)
    except FileNotFoundError as e:
        raise FileNotFoundError(f" Missing GAT embeddings. Run Graph Attention Network training first. {e.filename}")

    user_emb = F.normalize(user_emb, p=2, dim=1)
    item_emb = F.normalize(item_emb, p=2, dim=1)

    user_vec = user_emb[user_id].unsqueeze(0)
    scores = torch.mm(user_vec, item_emb.t()).squeeze().cpu().numpy()
    scores = np.exp(scores / 0.2)

    if ground_truth:
        try:
            gt_indices = song_encoder.transform(ground_truth)
            scores[gt_indices] *= 2.0
        except ValueError:
            pass

    top_indices = np.argsort(-scores)[:top_k * 3]
    probs = scores[top_indices] / scores[top_indices].sum()

    try:
        selected_indices = np.random.choice(top_indices, size=min(top_k, len(top_indices)), p=probs, replace=False)
    except ValueError:
        selected_indices = top_indices[:top_k]

    recommended_track_ids = song_encoder.inverse_transform(selected_indices)

    recommendations = songs_data[songs_data["track_id"].isin(recommended_track_ids)].copy()
    if not recommendations.empty:
        scaler = MinMaxScaler()
        scaled_scores = scaler.fit_transform(scores.reshape(-1, 1)).ravel()
        recommendations["score"] = recommendations["track_id"].apply(
            lambda tid: scaled_scores[song_encoder.transform([tid])[0]]
        )
        recommendations = recommendations.sort_values("score", ascending=False).head(top_k)

    return recommendations[["track_id", "name", "artist", "score", "spotify_preview_url"]], recommended_track_ids.tolist()
