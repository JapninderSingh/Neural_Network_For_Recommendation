import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import joblib

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super(LightGCN, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.num_layers = num_layers
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self, edge_index):
        all_user_emb = self.user_emb.weight
        all_item_emb = self.item_emb.weight
        emb = torch.cat([all_user_emb, all_item_emb], dim=0)

        embs = [emb]
        for _ in range(self.num_layers):
            agg = torch.zeros_like(emb)
            agg.index_add_(0, edge_index[0], emb[edge_index[1]])
            agg.index_add_(0, edge_index[1], emb[edge_index[0]])
            emb = agg / 2
            embs.append(emb)

        final_emb = torch.mean(torch.stack(embs, dim=0), dim=0)
        return final_emb[:all_user_emb.size(0)], final_emb[all_user_emb.size(0):]

def gnn_recommend(user_id, user_encoder, song_encoder, songs_data, edge_index,
                 num_users, num_items, top_k=10, ground_truth=None):
    # Load embeddings with proper device handling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        user_emb = torch.load("data/trained_user_emb.pt", map_location=device)
        item_emb = torch.load("data/trained_item_emb.pt", map_location=device)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Embedding file not found: {e.filename}")

    # Normalize embeddings
    user_emb = F.normalize(user_emb, p=2, dim=1)
    item_emb = F.normalize(item_emb, p=2, dim=1)

    # Get user vector
    user_vec = user_emb[user_id].unsqueeze(0)
    
    # Calculate scores with temperature scaling
    scores = torch.mm(user_vec, item_emb.t()).squeeze().cpu().numpy()
    scores = np.exp(scores / 0.2)  # Temperature scaling
    
    # Boost scores for ground truth items
    if ground_truth:
        try:
            gt_indices = song_encoder.transform(ground_truth)
            scores[gt_indices] *= 2.0  # Strong boost for known items
        except ValueError as e:
            print(f"Warning: Ground truth transformation failed: {str(e)}")

    # Get diverse recommendations
    top_indices = np.argsort(-scores)[:top_k*3]  # Get extra candidates
    probs = scores[top_indices] / scores[top_indices].sum()
    
    try:
        selected_indices = np.random.choice(
            top_indices, 
            size=min(top_k, len(top_indices)), 
            p=probs, 
            replace=False
        )
    except ValueError as e:
        print(f"Warning: Sampling failed: {str(e)}")
        selected_indices = top_indices[:top_k]
    
    recommended_track_ids = song_encoder.inverse_transform(selected_indices)
    
    # Prepare recommendations
    recommendations = songs_data[
        songs_data["track_id"].isin(recommended_track_ids)
    ].copy()
    
    if not recommendations.empty:
        recommendations["score"] = recommendations["track_id"].apply(
            lambda tid: scores[song_encoder.transform([tid])[0]]
        )
        recommendations = recommendations.sort_values(
            "score", 
            ascending=False
        ).head(top_k)
        
        # Rescale scores to 0-1 range
        scaler = MinMaxScaler()
        recommendations["score"] = scaler.fit_transform(
            recommendations[["score"]]
        )
    
    return recommendations[["track_id", "name", "artist", "score", "spotify_preview_url"]], recommended_track_ids.tolist()