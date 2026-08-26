import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.config import PROCESSED_DATA_PATH


# Path to cache the similarity matrix
SIMILARITY_MATRIX_PATH = PROCESSED_DATA_PATH / "content_similarity.joblib"


def build_genre_matrix(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode genres for each movie.

    Example:
        Toy Story -> Animation|Children's|Comedy
        Result:   [1, 0, 1, 1, 0, 0, ...]  (Animation=1, Comedy=1, Children's=1)

    Returns:
        DataFrame with movieId as index and genre columns as binary features.
    """
    genre_dummies = movies_df["genres"].str.get_dummies(sep="|")
    genre_dummies.index = movies_df["movieId"]
    return genre_dummies


def compute_similarity_matrix(genre_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cosine similarity between all movie pairs based on genre vectors.

    Returns:
        DataFrame of shape (n_movies, n_movies) with movieIds as index/columns.
    """
    similarity = cosine_similarity(genre_matrix)
    return pd.DataFrame(
        similarity,
        index=genre_matrix.index,
        columns=genre_matrix.index
    )


def get_similarity_matrix(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load cached similarity matrix or compute and cache it.
    """
    if SIMILARITY_MATRIX_PATH.exists():
        print("Loading cached content similarity matrix...")
        return joblib.load(SIMILARITY_MATRIX_PATH)

    print("Computing content similarity matrix...")
    genre_matrix = build_genre_matrix(movies_df)
    sim_matrix = compute_similarity_matrix(genre_matrix)

    joblib.dump(sim_matrix, SIMILARITY_MATRIX_PATH)
    print(f"Cached similarity matrix to {SIMILARITY_MATRIX_PATH}")
    return sim_matrix


def get_content_scores(
    user_id: int,
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    sim_matrix: pd.DataFrame,
) -> dict:
    """
    For a given user, compute a content-based score for every unrated movie.

    How it works:
        1. Find all movies this user has rated (and their ratings).
        2. For each unrated movie, compute a weighted average of its similarity
           to the user's rated movies, weighted by the user's actual ratings.

    Returns:
        dict of {movieId: content_score}
    """
    # Get this user's rated movies and their ratings
    user_ratings = train_df[train_df["userId"] == user_id]
    rated_movie_ids = user_ratings["movieId"].tolist()
    ratings_dict = dict(zip(user_ratings["movieId"], user_ratings["rating"]))

    # All movie IDs in the similarity matrix
    all_movie_ids = sim_matrix.index.tolist()
    unrated_movies = [m for m in all_movie_ids if m not in rated_movie_ids]

    scores = {}
    for movie_id in unrated_movies:
        if movie_id not in sim_matrix.index:
            continue

        # Weighted sum: similarity to rated movies × user's rating for them
        numerator = 0.0
        denominator = 0.0
        for rated_id, rating in ratings_dict.items():
            if rated_id in sim_matrix.columns:
                sim = sim_matrix.loc[movie_id, rated_id]
                numerator += sim * rating
                denominator += abs(sim)

        if denominator > 0:
            scores[movie_id] = numerator / denominator
        else:
            scores[movie_id] = 0.0

    return scores
