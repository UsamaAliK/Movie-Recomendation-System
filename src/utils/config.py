from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
RAW_DATA_PATH = BASE_DIR / "data/raw"
PROCESSED_DATA_PATH = BASE_DIR / "data/processed"
NOTEBOOKS_PATH = BASE_DIR / "notebooks"
# Ensure directories exist
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_PATH.mkdir(parents=True, exist_ok=True)
# File paths
RATINGS_FILE = RAW_DATA_PATH / "ratings.dat"
MOVIES_FILE = RAW_DATA_PATH / "movies.dat"
RATING_CSV_FILE = PROCESSED_DATA_PATH / "ratings.csv"
MOVIES_CSV_FILE = PROCESSED_DATA_PATH / "movies.csv"
# Column names
RATINGS_COLUMNS = ['userId', 'movieId', 'rating', 'timestamp']
MOVIES_COLUMNS = ['movieId', 'title', 'genres']
# Data types
RATINGS_DTYPE = {
    'userId': 'int32',
    'movieId': 'int32',
    'rating': 'float32',
    'timestamp': 'int64'
}
MOVIES_DTYPE = {
    'movieId': 'int32',
    'title': 'string',
    'genres': 'string'
}
# Separator
SEPARATOR = '::'
# Encoding
ENCODING = 'latin-1'

