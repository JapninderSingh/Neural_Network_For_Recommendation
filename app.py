
import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import torch
import joblib

from hybrid_recommendations import HybridRecommenderSystem
from content_based import content_recommendation
from collaborative_based import collaborative_recommendation
from gnn_model import gnn_recommend
from gnn_model_sage import gnn_recommend_sage
from gnn_model_gat import gnn_recommend_gat
from model_comparison_chart import display_model_comparison


os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNING"] = "true"
st.set_page_config(layout="wide", page_title="Neural Network For Recommender System")

@st.cache_data
def load_files():
    songs_data = pd.read_csv("data/cleaned_data.csv")
    transformed_data = load_npz("data/transformed_data.npz")
    interaction_matrix = load_npz("data/interaction_matrix.npz")
    track_ids = np.load("data/track_ids.npy", allow_pickle=True)
    return songs_data, transformed_data, interaction_matrix, track_ids

@st.cache_resource
def get_graph_data():
    try:
        edge_index = torch.load("data/edge_index.pt", map_location='cpu')
        num_users = int(torch.load("data/num_users.pt", map_location='cpu'))
        num_items = int(torch.load("data/num_items.pt", map_location='cpu'))
        user_encoder = joblib.load("data/user_encoder.pkl")
        song_encoder = joblib.load("data/song_encoder.pkl")
        return edge_index, num_users, num_items, user_encoder, song_encoder
    except Exception as e:
        st.error(f"Error loading GNN data: {str(e)}")
        st.stop()

def display_metrics(evaluation_metrics):
    st.subheader("Recommendation Quality")
    col1, col2 = st.columns(2)
    col1.progress(evaluation_metrics['Recall@K'], text=f"Taste Coverage: {evaluation_metrics['Recall@K']:.0%}")
    col1.caption("**Recall@K**: Percentage of your liked songs that appear in recommendations.")
    col2.progress(evaluation_metrics['NDCG@K'], text=f"Ranking Quality: {evaluation_metrics['NDCG@K']:.2f}")
    col2.caption("**NDCG@K**: How well recommendations are ranked (0-1 scale).")

def display_visualizations(recommendations):
    st.subheader("Recommendation Patterns")
    tab1, tab2 = st.tabs(["Artist Frequency", "Score Distribution"])
    with tab1:
        artist_counts = recommendations['artist'].value_counts().head(10)
        if not artist_counts.empty:
            st.bar_chart(artist_counts, color="#1DB954")
        else:
            st.warning("No artist data available")
    with tab2:
        if 'score' in recommendations:
            st.area_chart(recommendations.set_index('name')['score'], color="#1ED760")
        else:
            st.warning("No score data available")

def log_metrics(user_id, model, recall, ndcg, k, gt_len):
    df = pd.DataFrame([{
        "user_id": user_id,
        "model": model,
        "Recall@K": recall,
        "NDCG@K": ndcg,
        "K": k,
        "Ground_Truth_Size": gt_len
    }])
    log_path = "metrics_log.csv"
    if os.path.exists(log_path):
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, mode='w', header=True, index=False)

def main():
    st.title("Song Recommender System")
    songs_data, transformed_data, interaction_matrix, track_ids = load_files()

    st.sidebar.header("Recommendation Options")
    filtering_type = st.sidebar.selectbox(
        'Algorithm:',
        ['Content-Based Filtering', 'Collaborative Filtering',
         'Hybrid Recommender System', 'GNN (LightGCN)', 'GNN (GraphSAGE)', 'GNN (GAT)']
    )

    k = st.sidebar.slider('Number of Recommendations:', 5, 20, 10)
    show_debug = st.sidebar.checkbox("Show Technical Details", False)
    show_visuals = st.sidebar.checkbox("Show Recommendation Patterns", True)

    if 'GNN' in filtering_type:
        edge_index, num_users, num_items, user_encoder, song_encoder = get_graph_data()
        user_id = st.sidebar.selectbox("Select User ID:", user_encoder.classes_)
        show_metrics = True
    else:
        song_name = st.sidebar.text_input("Enter Song Name:", "Hips Don't Lie")
        artist_name = st.sidebar.text_input("Enter Artist Name:", "Shakira")
        show_metrics = False

    if st.sidebar.button('Get Recommendations', type="primary"):
        recommendations = pd.DataFrame()
        evaluation_metrics = {}
        with st.spinner(f'Generating {k} recommendations using {filtering_type}...'):
            try:
                if filtering_type == 'Content-Based Filtering':
                    recommendations = content_recommendation(song_name, artist_name, songs_data, transformed_data, k)
                elif filtering_type == 'Collaborative Filtering':
                    recommendations = collaborative_recommendation(song_name, artist_name, track_ids, songs_data, interaction_matrix, k)
                elif filtering_type == 'Hybrid Recommender System':
                    filtered_track_ids = set(track_ids)
                    filtered_songs_data = songs_data[songs_data["track_id"].isin(filtered_track_ids)].reset_index(drop=True)
                    filtered_indices = songs_data.index[songs_data["track_id"].isin(filtered_track_ids)].tolist()
                    filtered_transformed_data = transformed_data[filtered_indices]
                    recommender = HybridRecommenderSystem(k, weight_content_based=0.5)
                    recommendations = recommender.give_recommendations(
                        song_name, artist_name, filtered_songs_data, 
                        np.array(filtered_songs_data["track_id"]),
                        filtered_transformed_data, interaction_matrix
                    )
                elif 'GNN' in filtering_type:
                    encoded_user_id = user_encoder.transform([user_id])[0]
                    user_mask = edge_index[0] == encoded_user_id
                    gt_indices = (edge_index[1][user_mask] - num_users).unique()
                    gt_track_ids = [song_encoder.inverse_transform([idx])[0] for idx in gt_indices if 0 <= idx < num_items]

                    if filtering_type == 'GNN (LightGCN)':
                        recommendations, rec_track_ids = gnn_recommend(
                            encoded_user_id, user_encoder, song_encoder, songs_data,
                            edge_index, num_users, num_items, k, gt_track_ids
                        )
                    elif filtering_type == 'GNN (GraphSAGE)':
                        recommendations, rec_track_ids = gnn_recommend_sage(
                            encoded_user_id, user_encoder, song_encoder, songs_data,
                            edge_index, num_users, num_items, k, gt_track_ids
                        )
                    elif filtering_type == 'GNN (GAT)':
                        recommendations, rec_track_ids = gnn_recommend_gat(
                            encoded_user_id, user_encoder, song_encoder, songs_data,
                            edge_index, num_users, num_items, k, gt_track_ids
                        )

                    hits = len(set(rec_track_ids) & set(gt_track_ids))
                    recall = hits / len(gt_track_ids) if gt_track_ids else 0
                    relevance = [1 if tid in gt_track_ids else 0 for tid in rec_track_ids]
                    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance, 1))
                    idcg = sum(1 / np.log2(idx + 1) for idx in range(1, min(len(gt_track_ids), k) + 1))
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    evaluation_metrics = {"Recall@K": round(recall, 4), "NDCG@K": round(ndcg, 4)}
                    log_metrics(user_id, filtering_type, recall, ndcg, k, len(gt_track_ids))

                st.subheader(f"{filtering_type} Recommendations")
                if not recommendations.empty:
                    if show_metrics and evaluation_metrics:
                        display_metrics(evaluation_metrics)
                        st.markdown("---")
                    for _, row in recommendations.iterrows():
                        with st.container():
                            cols = st.columns([4, 1])
                            with cols[0]:
                                st.markdown(f"""**{row['name']}**  
*{row['artist']}*  
Match score: {row['score']:.6f}""")
                            with cols[1]:
                                if pd.notna(row.get('spotify_preview_url')):
                                    st.audio(row['spotify_preview_url'])
                            st.divider()
                    if show_visuals:
                        display_visualizations(recommendations)
                else:
                    st.warning("No recommendations could be generated.")
                if show_debug:
                    with st.expander("🔧 Debug Details"):
                        st.json({
                            "Recommendation Type": filtering_type,
                            "Top K": k,
                            "Recall@K": evaluation_metrics.get("Recall@K"),
                            "NDCG@K": evaluation_metrics.get("NDCG@K"),
                            "GT Items": len(gt_track_ids) if 'GNN' in filtering_type else "N/A"
                        })
            except Exception as e:
                st.error(f"Recommendation failed: {str(e)}")
                if show_debug:
                    st.exception(e)

    with st.expander("📊 Compare Models (Across Users)"):
        display_model_comparison()


    st.sidebar.header("Song Database")
    with st.sidebar.expander("Browse All Songs"):
        search_term = st.text_input("Search by song or artist:").strip().lower()
        filtered_songs = songs_data[
            songs_data["name"].str.lower().str.contains(search_term) |
            songs_data["artist"].str.lower().str.contains(search_term)
        ] if search_term else songs_data
        st.dataframe(filtered_songs[['name', 'artist']].drop_duplicates().sort_values('artist'),
                     height=300, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()
