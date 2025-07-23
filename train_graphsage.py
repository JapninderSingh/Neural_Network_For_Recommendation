import torch
import torch.nn.functional as F
import random
import joblib
from gnn_model_sage import GraphSAGE
from graph_data_preparation import prepare_graph_data

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
        interaction_dict[u].add(i - num_users)

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

def train_graphsage():
    print(" Loading graph data...")
    edge_index, num_users, num_items, user_encoder, song_encoder = prepare_graph_data("data/User_Listening_History.csv")
    print(f" Loaded graph with {num_users} users and {num_items} items.")
    print(" Starting GraphSAGE training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = edge_index.to(device)
    model = GraphSAGE(num_users, num_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 31):
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

    torch.save(user_emb.detach().cpu(), "data/trained_user_emb_sage.pt")
    torch.save(item_emb.detach().cpu(), "data/trained_item_emb_sage.pt")
    joblib.dump(user_encoder, "data/user_encoder.pkl")
    joblib.dump(song_encoder, "data/song_encoder.pkl")
    print(" Trained GraphSAGE embeddings and encoders saved!")

if __name__ == "__main__":
    train_graphsage()
