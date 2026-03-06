import pandas as pd
from surprise import SVD

def recommend_for_user(
    model: SVD,
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    user_id: int,
    top_n: int = 10
):
    rated_movies = train_df[train_df["userId"] == user_id]["movieId"].tolist()
    all_movies = movies_df["movieId"].unique()
    movies_to_predict = [m for m in all_movies if m not in rated_movies]

    predictions = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in movies_to_predict]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:top_n]

    recommended_movies = []
    for movie_id, rating in top_predictions:
        title = movies_df[movies_df["movieId"] == movie_id]["title"].values[0]
        recommended_movies.append({
            "title": title,
            "predicted_rating": rating
        })

    return pd.DataFrame(recommended_movies)