import streamlit as st
import pandas as pd
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data():
    cleaned_data_path = "data/cleaned_data.csv"
    songs_data = pd.read_csv(cleaned_data_path)
    return songs_data

songs_data = load_data()

# Content-Based Filtering
def content_based_recommendation(song_name, artist_name, songs_data, k):
    # Extract features and calculate similarity
    features = songs_data[["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]]
    song_idx = songs_data[(songs_data["name"].str.lower() == song_name) & (songs_data["artist"].str.lower() == artist_name)].index[0]
    similarities = cosine_similarity([features.iloc[song_idx]], features)
    recommended_indices = np.argsort(similarities[0])[::-1][1:k+1]
    return songs_data.iloc[recommended_indices]

# Hybrid Recommender System
class HybridRecommenderSystem:
    def __init__(self, number_of_recommendations, weight_content_based):
        self.number_of_recommendations = number_of_recommendations
        self.weight_content_based = weight_content_based

    def give_recommendations(self, song_name, artist_name, songs_data, interaction_matrix, k):
        # Generate content-based recommendations
        content_based_recs = content_based_recommendation(song_name, artist_name, songs_data, k)

        # Generate collaborative filtering recommendations (mock for demonstration)
        collaborative_recs = songs_data.sample(k)

        # Combine recommendations
        combined_recs = pd.concat([content_based_recs, collaborative_recs]).drop_duplicates().head(k)
        return combined_recs

# GNN-Based Recommender System
def gnn_based_recommendation(song_name, artist_name, songs_data, k):
    # Mock function for demonstration
    return songs_data.sample(k)

# Streamlit App
st.title('Welcome to the Spotify Song Recommender!')
st.write('### Enter the name of a song and the recommender will suggest similar songs 🎵🎧')

song_name = st.text_input('Enter a song name:').lower()
artist_name = st.text_input('Enter the artist name:').lower()
k = st.selectbox('How many recommendations do you want?', [5, 10, 15, 20], index=1)

filtering_type = st.selectbox('Select the recommender system:', ['Content-Based Filtering', 'Hybrid Recommender System', 'GNN-Based Recommender System'])

if st.button('Get Recommendations'):
    st.write(f"Generating recommendations using {filtering_type}...")

    if ((songs_data["name"].str.lower() == song_name) & (songs_data["artist"].str.lower() == artist_name)).any():
        if filtering_type == 'Content-Based Filtering':
            recommendations = content_based_recommendation(song_name, artist_name, songs_data, k)
        elif filtering_type == 'Hybrid Recommender System':
            recommender = HybridRecommenderSystem(k, 0.5)
            recommendations = recommender.give_recommendations(song_name, artist_name, songs_data, None, k)
        elif filtering_type == 'GNN-Based Recommender System':
            recommendations = gnn_based_recommendation(song_name, artist_name, songs_data, k)

        st.write(f"Recommendations for **{song_name}** by **{artist_name}**")
        for ind, recommendation in recommendations.iterrows():
            song = recommendation['name'].title()
            artist = recommendation['artist'].title()
            preview_url = recommendation.get('spotify_preview_url', None)
            if ind == 0:
                st.markdown("## Currently Playing")
                st.markdown(f"#### **{song}** by **{artist}**")
                if preview_url:
                    st.audio(preview_url)
                st.write('---')
            elif ind == 1:
                st.markdown("### Next Up 🎵")
                st.markdown(f"#### {ind}. **{song}** by **{artist}**")
                if preview_url:
                    st.audio(preview_url)
                st.write('---')
            else:
                st.markdown(f"#### {ind}. **{song}** by **{artist}**")
                if preview_url:
                    st.audio(preview_url)
                st.write('---')
    else:
        st.write(f"Sorry, we couldn't find {song_name} by {artist_name} in our database. Please try another song.")
