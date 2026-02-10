import pandas as pd
from src.utils.config import RATINGS_FILE, MOVIES_FILE,SEPARATOR,ENCODING,RATINGS_COLUMNS,MOVIES_COLUMNS

def load_ratings() -> pd.DataFrame:
    """
    Load MovieLens ratings from u.data
    Columns: userId, movieId, rating, timestamp
    """
    if not RATINGS_FILE.exists():
        raise FileNotFoundError(f"Ratings file not found: {RATINGS_FILE}")
    return pd.read_csv(RATINGS_FILE, sep=SEPARATOR,names=RATINGS_COLUMNS,engine='python',encoding=ENCODING)
def load_movies() -> pd.DataFrame:
    """Load MovieLens movies from u.item
    Columns: movieId, title, genres
    """
    if not MOVIES_FILE.exists():
        raise FileNotFoundError(f"Movies file not found: {MOVIES_FILE}")
    print(pd.read_csv(MOVIES_FILE, sep=SEPARATOR, names=MOVIES_COLUMNS, engine='python',encoding=ENCODING).head())
    return pd.read_csv(MOVIES_FILE, sep=SEPARATOR, names=MOVIES_COLUMNS, engine='python',encoding=ENCODING)
if __name__ == "__main__":
    # Load ratings and movies
    ratings = load_ratings()
    print("Ratings head:")
    print(ratings.head())

    movies = load_movies()
    print("Movies head:")
    print(movies.head())