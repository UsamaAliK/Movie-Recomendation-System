import numpy as np
import pandas as pd


def compute_alpha(user_id: int, train_df: pd.DataFrame, min_ratings: int = 20, max_ratings: int = 100) -> float:
    """
    Dynamically compute the collaborative filtering weight (alpha) based on
    how many ratings the user has provided.

    - Users with >= max_ratings ratings: alpha = 0.9 (trust SVD heavily)
    - Users with <= min_ratings ratings: alpha = 0.1 (trust content-based heavily)
    - In between: linear interpolation

    Returns alpha between 0.1 and 0.9.
    """
    num_ratings = len(train_df[train_df["userId"] == user_id])

    if num_ratings <= min_ratings:
        return 0.1
    if num_ratings >= max_ratings:
        return 0.9

    alpha = (num_ratings - min_ratings) / (max_ratings - min_ratings)
    return round(0.1 + alpha * 0.8, 2)


def hybrid_recommend(
    user_id: int,
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    svd_model,
    sim_matrix: pd.DataFrame,
    top_n: int = 10,
):
    """
    Generate hybrid recommendations blending SVD and content-based scores.

    For each unrated movie:
        final_score = alpha * svd_score + (1 - alpha) * content_score
    """
    from src.model.content_based import get_content_scores

    alpha = compute_alpha(user_id, train_df)
    print(f"  Alpha for user {user_id}: {alpha} (collaborative={alpha}, content={1 - alpha})")

    # SVD scores
    rated_movies = train_df[train_df["userId"] == user_id]["movieId"].tolist()
    all_movies = movies_df["movieId"].unique()
    unrated_movies = [m for m in all_movies if m not in rated_movies]

    svd_scores = {}
    for movie_id in unrated_movies:
        svd_scores[movie_id] = svd_model.predict(user_id, movie_id).est

    # Content-based scores
    content_scores = get_content_scores(user_id, train_df, movies_df, sim_matrix)

    # Normalize both score sets to [0, 1] range
    svd_values = np.array(list(svd_scores.values()))
    content_values = np.array(list(content_scores.values()))

    svd_min, svd_max = svd_values.min(), svd_values.max()
    content_min, content_max = content_values.min(), content_values.max()

    def normalize(val, vmin, vmax):
        if vmax - vmin == 0:
            return 0.5
        return (val - vmin) / (vmax - vmin)

    # Blend
    final_scores = []
    for movie_id in unrated_movies:
        svd_norm = normalize(svd_scores.get(movie_id, 0), svd_min, svd_max)
        content_norm = normalize(content_scores.get(movie_id, 0), content_min, content_max)

        blended = alpha * svd_norm + (1 - alpha) * content_norm
        final_scores.append((movie_id, blended))

    final_scores.sort(key=lambda x: x[1], reverse=True)
    top = final_scores[:top_n]

    results = []
    for movie_id, score in top:
        title = movies_df[movies_df["movieId"] == movie_id]["title"].values[0]
        results.append({"title": title, "hybrid_score": round(score, 4)})

    return pd.DataFrame(results), alpha
