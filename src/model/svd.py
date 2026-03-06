from pathlib import Path
import pandas as pd
from surprise import SVD,Dataset,Reader
from src.preprocessing.load_data import load_data
from src.utils.config import CLEANED_DATA_DIR, RATINGS_COLUMNS, RATING_CSV_FILE

import joblib

def load_ratings(file_path: Path=RATING_CSV_FILE) -> pd.DataFrame:
    ratings = load_data(file_path,RATINGS_COLUMNS,SEPARATOR,ENCODING)
    return ratings

def train_svd(rating_df: pd.DataFrame):
    reader = Reader(rating_scale=(rating_df['rating'].min(), rating_df['rating'].max()))
    data = Dataset.load_from_df(rating_df[['userId','movieId','rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    return model

def save_model(model: SVD, file_path: Path):
    joblib.dump(model, file_path)

def load_model(file_path: Path) -> SVD:
    return joblib.load(file_path)