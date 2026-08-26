import streamlit as st
import pandas as pd

from src.model.svd import load_model
from src.model.hybrid import hybrid_recommend, compute_alpha
from src.model.content_based import get_similarity_matrix, get_content_scores
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


def recommend_from_picks(picked_movies, movies_df, sim_matrix, top_n=10):
    """
    Given user's picked movies and their ratings, find similar movies
    using the content-based similarity matrix.
    """
    all_movie_ids = movies_df["movieId"].unique()
    picked_ids = [m["movieId"] for m in picked_movies]
    picked_ratings = {m["movieId"]: m["rating"] for m in picked_movies}

    scores = {}
    for movie_id in all_movie_ids:
        if movie_id in picked_ids:
            continue
        if movie_id not in sim_matrix.index:
            continue

        numerator = 0.0
        denominator = 0.0
        for pid, pr in picked_ratings.items():
            if pid in sim_matrix.columns:
                sim = sim_matrix.loc[movie_id, pid]
                numerator += sim * pr
                denominator += abs(sim)

        if denominator > 0:
            scores[movie_id] = numerator / denominator

    sorted_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for movie_id, score in sorted_movies:
        title = movies_df[movies_df["movieId"] == movie_id]["title"].values[0]
        genres = movies_df[movies_df["movieId"] == movie_id]["genres"].values[0]
        results.append({
            "title": title,
            "genres": genres,
            "score": round(score, 4),
        })
    return pd.DataFrame(results)


ratings, movies, model, sim_matrix = load_data()

st.title("Movie Recommendation System")
st.caption("Pick movies you like, and we'll recommend similar ones")

# --- Session state ---
if "picked" not in st.session_state:
    st.session_state.picked = []

# --- Sidebar ---
st.sidebar.header("Your Picks")
st.sidebar.write(f"**{len(st.session_state.picked)}** movies selected")
top_n = st.sidebar.slider("Number of recommendations", min_value=5, max_value=30, value=10)

if st.session_state.picked:
    st.sidebar.divider()
    st.sidebar.subheader("Selected Movies")
    for i, m in enumerate(st.session_state.picked):
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(f"{m['title']}")
        col2.write(f"{'⭐' * m['rating']}")
    if st.sidebar.button("Clear all picks", type="secondary"):
        st.session_state.picked = []
        st.rerun()

# --- Tabs ---
tab_pick, tab_rec, tab_eval = st.tabs(["Pick Movies", "Get Recommendations", "Model Evaluation"])

with tab_pick:
    st.subheader("Pick movies you like")

    movie_titles = movies["title"].tolist()
    selected_title = st.selectbox(
        "Type to search and select a movie",
        options=movie_titles,
        index=None,
        placeholder="Start typing a movie name...",
    )

    if selected_title:
        row = movies[movies["title"] == selected_title].iloc[0]
        already_picked = any(p["movieId"] == row["movieId"] for p in st.session_state.picked)

        if already_picked:
            st.warning(f"**{row['title']}** is already in your picks.")
        else:
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                st.write(f"**{row['title']}** — _{row['genres']}_")
            with col2:
                rating = st.slider("Rating", min_value=1, max_value=5, value=3, key="new_rating")
            with col3:
                if st.button("Add to picks", type="primary"):
                    st.session_state.picked.append({
                        "movieId": row["movieId"],
                        "title": row["title"],
                        "rating": rating,
                    })
                    st.rerun()

    if st.session_state.picked:
        st.divider()
        st.subheader("Your Picks")
        for i, m in enumerate(st.session_state.picked):
            col1, col2, col3 = st.columns([5, 1, 1])
            with col1:
                st.write(f"**{i+1}. {m['title']}**")
            with col2:
                st.write(f"{'⭐' * m['rating']}")
            with col3:
                if st.button("Remove", key=f"remove_{m['movieId']}"):
                    st.session_state.picked = [p for p in st.session_state.picked if p["movieId"] != m["movieId"]]
                    st.rerun()

with tab_rec:
    if not st.session_state.picked:
        st.info("Go to the **Pick Movies** tab and select some movies first.")
    else:
        with st.spinner("Finding recommendations..."):
            recs = recommend_from_picks(
                picked_movies=st.session_state.picked,
                movies_df=movies,
                sim_matrix=sim_matrix,
                top_n=top_n,
            )

        st.write(f"Based on your **{len(st.session_state.picked)}** picks:")

        for i, row in recs.iterrows():
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                st.write(f"**{i + 1}. {row['title']}**")
            with col2:
                st.caption(row["genres"])
            with col3:
                st.progress(row["score"])

with tab_eval:
    rmse, mae = load_evaluation()
    if rmse is not None:
        col1, col2 = st.columns(2)
        col1.metric("RMSE", f"{rmse:.4f}")
        col2.metric("MAE", f"{mae:.4f}")
        st.caption("Evaluated on the test set using SVD collaborative filtering only.")
    else:
        st.warning("Test file not found. Run the pipeline first.")
