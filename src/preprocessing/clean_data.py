import pandas as pd
import pathlib as path
from src.preprocessing.load_data import load_data
from src.utils.helpers import save_csv
from src.utils.config import CLEANED_DATA_DIR,RATINGS_FILE,MOVIES_FILE,SEPARATOR,ENCODING,RATINGS_COLUMNS,MOVIES_COLUMNS



def remove_duplicate_ratings(ratings:pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate ratings"""
    return ratings.drop_duplicates(subset=['userId', 'movieId'])


def remove_outlier_ratings(ratings:pd.DataFrame) -> pd.DataFrame:
    """Remove outlier ratings"""
    return ratings[ratings['rating'].between(1,5)]

def remove_duplicate_movies(movies:pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate movies"""
    return movies.drop_duplicates(subset=['movieId'])
    
def filter_inactive_users(ratings:pd.DataFrame,min_ratings:int=5) -> pd.DataFrame:
    """Filter out users with less than min_ratings"""
    user_counts = ratings['userId'].value_counts()
    return ratings[ratings['userId'].isin(user_counts[user_counts >= min_ratings].index)]

def filter_unpopular_movies(ratings:pd.DataFrame,min_ratings:int=5) -> pd.DataFrame:
    """Filter out movies with less than min_ratings"""
    movie_counts = ratings['movieId'].value_counts()
    return ratings[ratings['movieId'].isin(movie_counts[movie_counts >= min_ratings].index)]

def clean_ratings(ratings:pd.DataFrame) -> pd.DataFrame:
    """Clean the ratings dataframe"""
    ratings = remove_duplicate_ratings(ratings)
    ratings = remove_outlier_ratings(ratings)
    ratings = filter_inactive_users(ratings)
    ratings = filter_unpopular_movies(ratings)
    return ratings

def clean_movies(movies:pd.DataFrame) -> pd.DataFrame:
    """Clean the movies dataframe"""
    movies = remove_duplicate_movies(movies)
    return movies

def clean_data():
    """Clean the data"""
    ratings = load_data(RATINGS_FILE,RATINGS_COLUMNS,SEPARATOR,ENCODING)
    movies = load_data(MOVIES_FILE,MOVIES_COLUMNS,SEPARATOR,ENCODING)
    ratings = clean_ratings(ratings)
    movies = clean_movies(movies)
    save_csv(ratings, CLEANED_DATA_DIR, "ratings.csv")
    save_csv(movies, CLEANED_DATA_DIR, "movies.csv")

