import pandas as pd
import dask.dataframe as dd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# File paths
track_ids_save_path = "data/track_ids.npy"
track_id_to_idx_path = "data/track_id_to_idx.npy"
filtered_data_save_path = "data/collab_filtered_data.csv"
interaction_matrix_save_path = "data/interaction_matrix.npz"
songs_data_path = "data/cleaned_data.csv"
user_listening_history_data_path = "data/User Listening History.csv"

def filter_songs_data(songs_data: pd.DataFrame, track_ids: list, save_df_path: str) -> pd.DataFrame:
    filtered_data = songs_data[songs_data["track_id"].isin(track_ids)]
    filtered_data = filtered_data.sort_values(by="track_id").reset_index(drop=True)
    filtered_data.to_csv(save_df_path, index=False)
    return filtered_data

def save_sparse_matrix(matrix: csr_matrix, file_path: str) -> None:
    save_npz(file_path, matrix)

def create_interaction_matrix(history_data: dd.DataFrame, track_ids_save_path, save_matrix_path) -> csr_matrix:
    df = history_data.copy()
    df['playcount'] = df['playcount'].astype(np.float64)
    df = df.categorize(columns=['user_id', 'track_id'])

    user_mapping = df['user_id'].cat.codes
    track_mapping = df['track_id'].cat.codes

    track_ids = df['track_id'].cat.categories.values
    np.save(track_ids_save_path, track_ids, allow_pickle=True)

    df = df.assign(user_idx=user_mapping, track_idx=track_mapping)
    interaction_matrix = df.groupby(['track_idx', 'user_idx'])['playcount'].sum().reset_index().compute()

    row_indices = interaction_matrix['track_idx']
    col_indices = interaction_matrix['user_idx']
    values = interaction_matrix['playcount']

    n_tracks = row_indices.max() + 1
    n_users = col_indices.max() + 1

    matrix = csr_matrix((values, (row_indices, col_indices)), shape=(n_tracks, n_users))
    save_sparse_matrix(matrix, save_matrix_path)

    #  Save track_id → row_index mapping
    track_idx_to_track_id = pd.Series(track_ids)[interaction_matrix['track_idx']]
    track_id_to_idx_dict = dict(zip(track_idx_to_track_id, interaction_matrix['track_idx']))
    np.save(track_id_to_idx_path, track_id_to_idx_dict, allow_pickle=True)

    return matrix

def collaborative_recommendation(song_name, artist_name, track_ids, songs_data, interaction_matrix, k=5):
    song_name = song_name.lower()
    artist_name = artist_name.lower()

    song_row = songs_data.loc[
        (songs_data["name"].str.lower() == song_name) &
        (songs_data["artist"].str.lower() == artist_name)
    ]

    if song_row.empty:
        raise ValueError(f" Song '{song_name}' by '{artist_name}' not found in dataset.")

    input_track_id = song_row['track_id'].values.item()

    if not os.path.exists(track_id_to_idx_path):
        raise FileNotFoundError(" track_id_to_idx.npy not found. Run interaction matrix generation first.")

    track_id_to_idx = np.load(track_id_to_idx_path, allow_pickle=True).item()

    if input_track_id not in track_id_to_idx:
        raise ValueError(f" Track ID '{input_track_id}' not found in filtered interaction matrix.")

    ind = track_id_to_idx[input_track_id]

    # Ensure input_array is 2D for cosine similarity
    input_array = interaction_matrix[ind, :].reshape(1, -1)

    similarity_scores = cosine_similarity(input_array, interaction_matrix)

    recommendation_indices = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    recommendation_indices = recommendation_indices[recommendation_indices != ind][:k]

    recommendation_track_ids = track_ids[recommendation_indices]
    top_scores = similarity_scores.ravel()[recommendation_indices]

    scores_df = pd.DataFrame({
        "track_id": recommendation_track_ids.tolist(),
        "score": top_scores
    })

    #  KEEP the score column for Streamlit display
    top_k_songs = (
        songs_data[songs_data["track_id"].isin(recommendation_track_ids)]
        .merge(scores_df, on="track_id")
        .sort_values(by="score", ascending=False)
        .reset_index(drop=True)
    )

    return top_k_songs

def main():
    user_data = dd.read_csv(user_listening_history_data_path)
    unique_track_ids = user_data["track_id"].unique().compute().tolist()

    songs_data = pd.read_csv(songs_data_path)
    filter_songs_data(songs_data, unique_track_ids, filtered_data_save_path)
    create_interaction_matrix(user_data, track_ids_save_path, interaction_matrix_save_path)

if __name__ == "__main__":
    main()
