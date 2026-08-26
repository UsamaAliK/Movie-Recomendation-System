import streamlit as st
import pandas as pd

from src.model.svd import load_model
from src.model.hybrid import hybrid_recommend, compute_alpha
from src.model.content_based import get_similarity_matrix
from src.utils.config import (
    RATING_CSV_FILE,
    MOVIES_CSV_FILE,
    MODEL_PATH,
    CLEANED_DATA_DIR,
    TEST_FILE,
)
from src.evaluation.evaluation import evaluate_svd

st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_resource
def load_data():
    ratings = pd.read_csv(RATING_CSV_FILE)
    movies = pd.read_csv(MOVIES_CSV_FILE)
    model = load_model(MODEL_PATH)
    sim_matrix = get_similarity_matrix(movies)
    return ratings, movies, model, sim_matrix

@st.cache_data
def load_evaluation():
    test_file = CLEANED_DATA_DIR / TEST_FILE
    if not test_file.exists():
        return None, None
    ratings = pd.read_csv(RATING_CSV_FILE)
    model = load_model(MODEL_PATH)
    test_df = pd.read_csv(test_file, dtype={"rating": "float32"})
    return evaluate_svd(model, test_df)


ratings, movies, model, sim_matrix = load_data()

st.title("Movie Recommendation System")
st.caption("Hybrid model: Collaborative Filtering (SVD) + Content-Based (Genre Similarity)")

# --- Sidebar ---
st.sidebar.header("Settings")
user_id = st.sidebar.number_input(
    "User ID",
    min_value=int(ratings["userId"].min()),
    max_value=int(ratings["userId"].max()),
    value=1,
    step=1,
)
top_n = st.sidebar.slider("Number of recommendations", min_value=5, max_value=30, value=10)

# --- User Info ---
num_ratings = len(ratings[ratings["userId"] == user_id])
alpha = compute_alpha(user_id, ratings)

st.sidebar.divider()
st.sidebar.metric("User's Ratings", num_ratings)
st.sidebar.metric("Alpha (CF weight)", f"{alpha}")
st.sidebar.metric("Content weight", f"{1 - alpha}")

# --- Main Content ---
tab_rec, tab_rated, tab_eval = st.tabs(["Recommendations", "User's Ratings", "Model Evaluation"])

with tab_rec:
    with st.spinner("Generating recommendations..."):
        recs, alpha = hybrid_recommend(
            user_id=user_id,
            train_df=ratings,
            movies_df=movies,
            svd_model=model,
            sim_matrix=sim_matrix,
            top_n=top_n,
        )

    st.info(f"Alpha = **{alpha}** — Collaborative Filtering: **{alpha}**, Content-Based: **{1 - alpha}**")

    for i, row in recs.iterrows():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**{i + 1}. {row['title']}**")
        with col2:
            st.progress(row["hybrid_score"])

with tab_rated:
    user_ratings = ratings[ratings["userId"] == user_id].merge(movies, on="movieId")
    user_ratings = user_ratings.sort_values("rating", ascending=False)
    st.write(f"**{len(user_ratings)}** rated movies by User {user_id}")
    st.dataframe(
        user_ratings[["title", "genres", "rating"]].reset_index(drop=True),
        use_container_width=True,
    )

with tab_eval:
    rmse, mae = load_evaluation()
    if rmse is not None:
        col1, col2 = st.columns(2)
        col1.metric("RMSE", f"{rmse:.4f}")
        col2.metric("MAE", f"{mae:.4f}")
        st.caption("Evaluated on the test set using SVD collaborative filtering only.")
    else:
        st.warning("Test file not found. Run the pipeline first to generate train/test splits.")
