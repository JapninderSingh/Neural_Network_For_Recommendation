import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.preprocessing import MinMaxScaler

# File paths
songs_path = os.path.join("data", "Music_Info.csv")
track_ids_path = os.path.join("data", "track_ids.npy")
interaction_matrix_path = os.path.join("data", "interaction_matrix.npz")
transformed_features_path = os.path.join("data", "transformed_features.npz")

# Load songs.csv
try:
    songs_df = pd.read_csv(songs_path)
    print("Loaded songs.csv")
except FileNotFoundError:
    print(" songs.csv not found in the 'data' folder.")
    exit()

#  Save track_ids.npy
track_ids = songs_df['track_id'].values
np.save(track_ids_path, track_ids)
print(" Saved track_ids.npy")

# Generate interaction_matrix.npz (dummy play counts per track)
# Simulate interactions (e.g., 100 users × len(track_ids) matrix)
num_users = 100
num_tracks = len(track_ids)

np.random.seed(42)
interactions = np.random.poisson(lam=1.5, size=(num_users, num_tracks))
interaction_matrix = csr_matrix(interactions)
save_npz(interaction_matrix_path, interaction_matrix)
print(f" Saved interaction_matrix.npz ({num_users} users × {num_tracks} songs)")

# Generate transformed_features.npz (scaled audio features)
# Extract numeric columns for content-based filtering
non_feature_cols = ['track_id', 'name', 'artist']
feature_cols = [col for col in songs_df.columns if col not in non_feature_cols and songs_df[col].dtype != 'object']

if not feature_cols:
    print(" No numeric feature columns found for content-based filtering.")
    exit()

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(songs_df[feature_cols])
transformed_matrix = csr_matrix(scaled_features)
save_npz(transformed_features_path, transformed_matrix)
print(f" Saved transformed_features.npz ({len(feature_cols)} features scaled)")
