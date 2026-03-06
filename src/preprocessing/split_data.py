import pandas as pd
import pathlib as path
from src.utils.helpers import save_csv
from src.utils.config import CLEANED_DATA_DIR,TRAIN_FILE,TEST_FILE

def time_based_split(rating:pd.DataFrame,test_ratio:float=0.2,):
    rating_sorted = rating.sort_values(by=['userId','timestamp'])
    train_list=[]
    test_list=[]
    for user_id,group in rating_sorted.groupby('userId'):
        num_ratings=len(group)
        n_test=max(1,int(num_ratings*test_ratio))
        train_list.append(group.iloc[:n_test])
        test_list.append(group.iloc[n_test:])
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    return train_df,test_df

def split_data(test_ratio:float=0.2):
    ratings = load_data(CLEANED_DATA_DIR / "ratings.csv",RATINGS_COLUMNS,SEPARATOR,ENCODING)
    train_df,test_df = time_based_split(ratings,test_ratio)
    save_csv(train_df,CLEANED_DATA_DIR,TRAIN_FILE)
    save_csv(test_df,CLEANED_DATA_DIR,TEST_FILE)
if __name__ == "__main__":
    split_data(test_ratio=0.2)