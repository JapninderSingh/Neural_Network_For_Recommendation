
import os
import pandas as pd
import altair as alt
import streamlit as st

def display_model_comparison():
    if not os.path.exists("metrics_log.csv"):
        st.info("Run some GNN recommendations to see comparison charts.")
        return

    df = pd.read_csv("metrics_log.csv")

    st.subheader(" GNN Model Comparison (Per-User)")
    metric = st.selectbox("Select Metric to Compare:", ["Recall@K", "NDCG@K"])
    selected_models = st.multiselect("Select Models to Include:", df["model"].unique().tolist(), default=df["model"].unique().tolist())

    filtered = df[df["model"].isin(selected_models)]

    if filtered.empty:
        st.warning("No data for selected models.")
        return

    chart = alt.Chart(filtered).mark_line(point=True).encode(
        x=alt.X("user_id:N", title="User ID"),
        y=alt.Y(metric, scale=alt.Scale(domain=[0, 1]), title=metric),
        color=alt.Color("model:N", title="Model"),
        tooltip=["user_id", "model", metric]
    ).properties(width=800, height=400)

    st.altair_chart(chart, use_container_width=True)
