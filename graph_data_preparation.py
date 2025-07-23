import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import joblib

def prepare_graph_data(filepath="data/User_Listening_History.csv", max_users=5000, max_items=3000):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["user_id", "track_id"])

    # Keep only top users and top items
    top_users = df["user_id"].value_counts().nlargest(max_users).index
    top_items = df["track_id"].value_counts().nlargest(max_items).index
    df = df[df["user_id"].isin(top_users) & df["track_id"].isin(top_items)].copy()

    # Re-check after filtering
    actual_users = df["user_id"].nunique()
    actual_items = df["track_id"].nunique()

    # Encode
    user_encoder = LabelEncoder()
    song_encoder = LabelEncoder()
    df["user"] = user_encoder.fit_transform(df["user_id"])
    df["item"] = song_encoder.fit_transform(df["track_id"])

    # Build edge index
    edge_index = torch.tensor(df[["user", "item"]].values).T
    edge_index[1] += actual_users  # Shift item indices

    # Save files
    torch.save(edge_index, "data/edge_index.pt")
    torch.save(torch.tensor(actual_users), "data/num_users.pt")
    torch.save(torch.tensor(actual_items), "data/num_items.pt")
    joblib.dump(user_encoder, "data/user_encoder.pkl")
    joblib.dump(song_encoder, "data/song_encoder.pkl")

    return edge_index, actual_users, actual_items, user_encoder, song_encoder
