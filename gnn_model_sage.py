import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

class GraphSAGE(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=2):
        super(GraphSAGE, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.num_layers = num_layers
        self.linear = nn.Linear(embedding_dim * 2, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, edge_index):
        emb_users = self.user_emb.weight
        emb_items = self.item_emb.weight
        emb = torch.cat([emb_users, emb_items], dim=0)

        for _ in range(self.num_layers):
            agg = torch.zeros_like(emb)
            agg.index_add_(0, edge_index[0], emb[edge_index[1]])
            agg.index_add_(0, edge_index[1], emb[edge_index[0]])
            concat = torch.cat([emb, agg], dim=1)
            emb = F.relu(self.linear(concat))

        return emb[:emb_users.size(0)], emb[emb_users.size(0):]

def gnn_recommend_sage(user_id, user_encoder, song_encoder, songs_data, edge_index,
                       num_users, num_items, top_k=10, ground_truth=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        user_emb = torch.load("data/trained_user_emb_sage.pt", map_location=device)
        item_emb = torch.load("data/trained_item_emb_sage.pt", map_location=device)
    except FileNotFoundError as e:
        raise FileNotFoundError(f" Missing SAGE embeddings. Run GraphSAGE training first. {e.filename}")

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
        recommendations["score"] = recommendations["track_id"].apply(
            lambda tid: scores[song_encoder.transform([tid])[0]]
        )
        recommendations = recommendations.sort_values("score", ascending=False).head(top_k)
        scaler = MinMaxScaler()
        # Global normalization
        all_scaled_scores = scaler.fit_transform(scores.reshape(-1, 1)).ravel()
        recommendations["score"] = recommendations["track_id"].apply(
        lambda tid: all_scaled_scores[song_encoder.transform([tid])[0]]
)


    return recommendations[["track_id", "name", "artist", "score", "spotify_preview_url"]], recommended_track_ids.tolist()
