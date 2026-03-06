from surprise import accuracy
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
import pandas as pd


def evaluate_svd(model, test_df):
    
    reader = Reader(rating_scale=(test_df['rating'].min(), test_df['rating'].max()))
    
    data = Dataset.load_from_df(
        test_df[['userId','movieId','rating']],
        reader
    )
    
    testset = data.build_full_trainset().build_testset()

    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    return rmse, mae
