import pandas as pd
from src.preprocessing.clean_data import clean_data
from src.preprocessing.load_data import load_data
from src.preprocessing.split_data import split_data

from src.model.svd import train_svd, save_model, load_model
from src.model.hybrid import hybrid_recommend
from src.model.content_based import get_similarity_matrix
from src.evaluation.evaluation import evaluate_svd

from src.utils.config import (
    RATING_CSV_FILE,
    MOVIES_CSV_FILE,
    MODEL_PATH,
    RATINGS_COLUMNS,
    MOVIES_COLUMNS,
    TRAIN_FILE,
    TEST_FILE,
    CLEANED_DATA_DIR
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

    test_file = CLEANED_DATA_DIR / TEST_FILE
    if test_file.exists():
        print("\nEvaluating model...")
        test_df = pd.read_csv(test_file, dtype={"rating": "float32"})
        rmse, mae = evaluate_svd(model, test_df)
        print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    else:
        print("\nTest file not found, skipping evaluation.")

    user_id = 1

    print(f"\nGenerating hybrid recommendations for user {user_id}")

    sim_matrix = get_similarity_matrix(movies)

    recommendations, alpha = hybrid_recommend(
        user_id=user_id,
        train_df=ratings,
        movies_df=movies,
        svd_model=model,
        sim_matrix=sim_matrix,
        top_n=10
    )

    print(f"\nRecommended Movies (alpha={alpha}):")
    print(recommendations)


if __name__ == "__main__":
    main()