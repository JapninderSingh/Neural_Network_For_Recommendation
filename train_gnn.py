import torch
import torch.nn.functional as F
import numpy as np
from gnn_model import LightGCN
from graph_data_preparation import prepare_graph_data

# Load data
edge_index, num_users, num_items, user_enc, song_enc = prepare_graph_data("data/User_Listening_History.csv")

# Shift item indices to create bipartite graph
edge_index[1] += num_users

# Initialize model
model = LightGCN(num_users, num_items)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Prepare positive pairs
user_ids = edge_index[0]
item_ids = edge_index[1] - num_users  # original item indices

def sample_negative(batch_size):
    return torch.randint(0, num_items, (batch_size,), dtype=torch.long)

# Training loop with BPR loss
model.train()
epochs = 200

for epoch in range(epochs):
    optimizer.zero_grad()
    user_emb, item_emb = model(edge_index)

    users = user_ids
    pos_items = item_ids
    neg_items = sample_negative(len(users))

    u_emb = user_emb[users]
    pos_emb = item_emb[pos_items]
    neg_emb = item_emb[neg_items]

    pos_scores = torch.sum(u_emb * pos_emb, dim=1)
    neg_scores = torch.sum(u_emb * neg_emb, dim=1)
    loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: BPR Loss = {loss.item():.4f}")

# Save final embeddings
torch.save(user_emb, "gnn/user_embeddings.pt")
torch.save(item_emb, "gnn/item_embeddings.pt")
