
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="EcoVision-AI Dashboard",
    page_icon="♻️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "predictions.db"

st.title("♻️ EcoVision-AI Analytics Dashboard")
st.caption("Live statistics from your AI waste classification model")

if not DB_PATH.exists():
    st.warning("Database not found. Start the backend and make a prediction first.")
    st.stop()

conn = sqlite3.connect(DB_PATH)

df = pd.read_sql_query(
    "SELECT * FROM predictions ORDER BY id DESC",
    conn
)

conn.close()

if df.empty:
    st.info("No predictions yet. Upload a waste image in EcoVision-AI first.")
    st.stop()

# Convert date
df["created_at"] = pd.to_datetime(df["created_at"])

# Dashboard metrics
total = len(df)
categories = df["category"].nunique()
avg_confidence = df["confidence"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("Total Images Analyzed", total)
col2.metric("Waste Categories Found", categories)
col3.metric("Average Confidence", f"{avg_confidence:.2f}%")

st.divider()

# Category counts
st.subheader("Waste Classification Statistics")

counts = (
    df["category"]
    .value_counts()
    .rename_axis("Category")
    .reset_index(name="Count")
)

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        counts,
        x="Category",
        y="Count",
        color="Category",
        title="Waste Category Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = px.pie(
        counts,
        names="Category",
        values="Count",
        hole=0.4,
        title="Waste Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)

# Recent predictions
st.subheader("Recent AI Predictions")

recent = df[["id", "category", "confidence", "created_at"]].head(20)

st.dataframe(
    recent,
    use_container_width=True,
    hide_index=True
)

# Export records
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Prediction History",
    data=csv,
    file_name="ecovision_predictions.csv",
    mime="text/csv"
)

if st.button("Refresh Dashboard"):
    st.rerun()

st.success("Dashboard connected to real AI prediction data!")
