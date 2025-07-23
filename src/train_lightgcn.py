import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import joblib
from graph_data_preparation import prepare_graph_data

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super(LightGCN, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.num_layers = num_layers
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, edge_index):
        all_user_emb = self.user_emb.weight
        all_item_emb = self.item_emb.weight
        emb = torch.cat([all_user_emb, all_item_emb], dim=0)

        for _ in range(self.num_layers):
            agg = torch.zeros_like(emb)
            agg.index_add_(0, edge_index[0], emb[edge_index[1]])
            agg.index_add_(0, edge_index[1], emb[edge_index[0]])
            emb = agg / 2

        user_final = emb[:all_user_emb.size(0)]
        item_final = emb[all_user_emb.size(0):]
        return user_final, item_final

def bpr_loss(user_emb, pos_emb, neg_emb):
    pos_scores = torch.sum(user_emb * pos_emb, dim=1)
    neg_scores = torch.sum(user_emb * neg_emb, dim=1)
    return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

def sample_pairs(edge_index, num_users, num_items, num_samples):
    user_item_pairs = edge_index.T.cpu().numpy()
    interaction_dict = {}

    for u, i in user_item_pairs:
        if u not in interaction_dict:
            interaction_dict[u] = set()
        interaction_dict[u].add(i - num_users)  # logical item index

    users, pos_items, neg_items = [], [], []

    for _ in range(num_samples):
        u = random.randint(0, num_users - 1)
        if u not in interaction_dict or not interaction_dict[u]:
            continue

        pos = random.choice(list(interaction_dict[u]))

        while True:
            neg = random.randint(0, num_items - 1)
            if neg not in interaction_dict[u]:
                break

        users.append(u)
        pos_items.append(pos)
        neg_items.append(neg)

    return torch.tensor(users), torch.tensor(pos_items), torch.tensor(neg_items)

def train_lightgcn():
    print(" Loading graph data...")
    edge_index, num_users, num_items, user_encoder, song_encoder = prepare_graph_data("data/User_Listening_History.csv")
    print(f" Loaded graph with {num_users} users and {num_items} items.")
    print(" Starting training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = edge_index.to(device)
    model = LightGCN(num_users, num_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 31):  # Increase epochs
        model.train()
        users, pos_items, neg_items = sample_pairs(edge_index, num_users, num_items, 2048)
        users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)

        optimizer.zero_grad()
        user_emb, item_emb = model(edge_index)

        loss = bpr_loss(
            user_emb[users],
            item_emb[pos_items],
            item_emb[neg_items]
        )
        loss.backward()
        optimizer.step()

        print(f" Epoch {epoch:02d} | BPR Loss: {loss.item():.4f}")

    torch.save(user_emb.detach().cpu(), "data/trained_user_emb.pt")
    torch.save(item_emb.detach().cpu(), "data/trained_item_emb.pt")
    joblib.dump(user_encoder, "data/user_encoder.pkl")
    joblib.dump(song_encoder, "data/song_encoder.pkl")
    print(" Trained embeddings and encoders saved!")

if __name__ == "__main__":
    train_lightgcn()
