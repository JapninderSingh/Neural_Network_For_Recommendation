import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

print(" hybrid_recommendations.py loaded correctly")


class HybridRecommenderSystem:

    def __init__(self, number_of_recommendations: int, weight_content_based: float):
        self.number_of_recommendations = number_of_recommendations
        self.weight_content_based = weight_content_based
        self.weight_collaborative = 1 - weight_content_based

    def __calculate_content_based_similarities(self, song_name, artist_name, songs_data, transformed_matrix):
        song_row = songs_data.loc[
            (songs_data["name"].str.lower() == song_name.lower()) &
            (songs_data["artist"].str.lower() == artist_name.lower())
        ]

        if song_row.empty:
            raise ValueError(f" Song '{song_name}' by '{artist_name}' not found in songs_data.")

        song_index = song_row.index[0]
        input_vector = transformed_matrix[song_index].reshape(1, -1)
        return cosine_similarity(input_vector, transformed_matrix)

    def __calculate_collaborative_filtering_similarities(self, song_name, artist_name, track_ids, songs_data, interaction_matrix):
        song_row = songs_data.loc[
            (songs_data["name"].str.lower() == song_name.lower()) &
            (songs_data["artist"].str.lower() == artist_name.lower())
        ]

        if song_row.empty:
            raise ValueError(f" Song '{song_name}' by '{artist_name}' not found in songs_data.")

        input_track_id = song_row['track_id'].values.item()

        # Load track_id_to_idx mapping
        track_id_to_idx_path = "data/track_id_to_idx.npy"
        if not os.path.exists(track_id_to_idx_path):
            raise FileNotFoundError(" track_id_to_idx.npy not found. Run interaction matrix generation first.")

        track_id_to_idx = np.load(track_id_to_idx_path, allow_pickle=True).item()

        if input_track_id not in track_id_to_idx:
            raise ValueError(f" Track ID '{input_track_id}' not found in filtered interaction matrix.")

        ind = track_id_to_idx[input_track_id]
        input_vector = interaction_matrix[ind, :].reshape(1, -1)
        return cosine_similarity(input_vector, interaction_matrix)

    def __normalize_similarities(self, similarity_scores):
        min_val = np.min(similarity_scores)
        max_val = np.max(similarity_scores)
        if max_val == min_val:
            return np.zeros_like(similarity_scores)
        return (similarity_scores - min_val) / (max_val - min_val)

    def __weighted_combination(self, content_scores, collaborative_scores):
        return (self.weight_content_based * content_scores) + (self.weight_collaborative * collaborative_scores)

    def give_recommendations(self, song_name, artist_name, songs_data, track_ids, transformed_matrix, interaction_matrix):
        content_sim = self.__calculate_content_based_similarities(song_name, artist_name, songs_data, transformed_matrix)
        collab_sim = self.__calculate_collaborative_filtering_similarities(song_name, artist_name, track_ids, songs_data, interaction_matrix)

        content_sim_norm = self.__normalize_similarities(content_sim)
        collab_sim_norm = self.__normalize_similarities(collab_sim)

        combined_scores = self.__weighted_combination(content_sim_norm, collab_sim_norm)

        # Get top recommendations (excluding the input song)
        indices = np.argsort(combined_scores.ravel())[-self.number_of_recommendations - 1:][::-1]
        input_song_row = songs_data.loc[
            (songs_data["name"].str.lower() == song_name.lower()) &
            (songs_data["artist"].str.lower() == artist_name.lower())
        ].index[0]

        indices = [i for i in indices if i != input_song_row][:self.number_of_recommendations]
        top_scores = combined_scores.ravel()[indices]
        top_track_ids = songs_data.iloc[indices]["track_id"].values

        scores_df = pd.DataFrame({
            "track_id": top_track_ids,
            "score": top_scores
        })

        top_k_songs = (
            songs_data[songs_data["track_id"].isin(top_track_ids)]
            .merge(scores_df, on="track_id")
            .sort_values(by="score", ascending=False)
            .reset_index(drop=True)
        )

        print(" Hybrid recommendations generated successfully.")
        return top_k_songs
