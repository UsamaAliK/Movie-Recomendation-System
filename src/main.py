import pandas as pd
from src.preprocessing.clean_data import clean_data
from src.preprocessing.load_data import load_data
from src.preprocessing.split_data import split_data

from src.model.svd import train_svd, save_model, load_model
from src.recommendation.recommend import recommend_for_user

from src.utils.config import (
    RATING_CSV_FILE,
    MOVIES_CSV_FILE,
    MODEL_PATH,
    RATINGS_COLUMNS,
    MOVIES_COLUMNS
)


def main():

    if not RATING_CSV_FILE.exists() or not MOVIES_CSV_FILE.exists():
        print("Cleaning data...")
        clean_data()

        print("Splitting data...")
        split_data()

    print("Loading cleaned data...")
    ratings = pd.read_csv(RATING_CSV_FILE)
    movies = pd.read_csv(MOVIES_CSV_FILE)

    if MODEL_PATH.exists():
        print("Loading saved SVD model...")
        model = load_model(MODEL_PATH)
    else:
        print("Training SVD model...")
        model = train_svd(ratings)
        print("Saving SVD model...")
        save_model(model, MODEL_PATH)

    user_id = 1

    print(f"Generating recommendations for user {user_id}")

    recommendations = recommend_for_user(
        model=model,
        train_df=ratings,
        movies_df=movies,
        user_id=user_id,
        top_n=10
    )

    print("\nRecommended Movies:")
    print(recommendations)


if __name__ == "__main__":
    main()