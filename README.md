#  Neural Network for Music Recommendation

A hybrid recommender system using Graph Neural Networks (GNN), collaborative filtering, and content-based filtering for personalized music recommendation. Built with Spotify track metadata and implicit feedback (listening history), and served through an interactive Streamlit app.

---

##  Features

-  Content-Based Filtering using TF-IDF + audio features
-  Collaborative Filtering using user-song interaction matrix
-  Hybrid Recommender: weighted fusion of content + collaborative similarity
-  Graph Neural Network (LightGCN) using implicit feedback
-  Personalized evaluation: Recall@K and NDCG@K
-  Interactive UI with search, filters, audio previews, and explanation features

---

##  Project Structure

```
.
├── app.py                        # Streamlit frontend
├── content_based.py             # Content-based logic
├── collaborative_based.py       # Collaborative filtering logic
├── hybrid_recommendations.py    # Weighted hybrid recommender
├── gnn_model.py                 # LightGCN model and recommendation
├── train_lightgcn.py            # GNN training script
├── data_cleaning.py             # Dataset cleaning logic
├── graph_data_preparation.py    # GNN data prep and encoding
├── generate_all_data_files.py   # Matrix and feature generation
├── generate_hybrid_transformed_data.py # Hybrid dataset transformation
├── data/                        # All CSVs, encoders, matrices, and embeddings
├── transformer.joblib           # Pre-trained ColumnTransformer
└── requirements.txt             # All required Python libraries
```

---

##  Installation

### Prerequisites

- Python 3.8+
- pip

###  Setup

1. Clone the repository:
   ```bash
   git clone https://git.cs.kent.ac.uk/jj469/neural_network_for_recommendation.git
   cd neural_network_for_recommendation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

##  Data Preparation & Training

### 1. Clean the dataset
```bash
python data_cleaning.py
```

### 2. Create interaction matrix and features
```bash
python generate_all_data_files.py
```

### 3. Train content transformer and save transformed data
```bash
python content_based.py
```

### 4. Prepare collaborative-filtered content data
```bash
python generate_hybrid_transformed_data.py
```

### 5. Prepare graph data for GNN
```bash
python graph_data_preparation.py
```

### 6. Train LightGCN model
```bash
python train_lightgcn.py
```

---

##  Running the App

```bash
streamlit run app.py
```

Then open your browser to:

- `http://localhost:8502` (if local)
- Or use your local IP (e.g., `http://192.168.1.42:8502`) for other devices on same Wi-Fi

---

##  Evaluation Metrics

The system computes:
- **Recall@K** – how many of a user's actual liked songs were recommended
- **NDCG@K** – quality of ranking based on ground-truth item positions

---

##  Data Used

- `Music_Info.csv`: Spotify audio features + metadata
- `User_Listening_History.csv`: Implicit feedback (play counts)
- Preprocessed sparse matrices, encoders, and embeddings stored under `/data/`

---

##  Author

**Japninder Singh**  


---

##  License

This project is for academic purposes under University of Kent coursework.
