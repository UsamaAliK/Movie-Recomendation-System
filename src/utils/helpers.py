from pathlib import Path
import pandas as pd

def save_csv(df: pd.DataFrame, folder: Path, filename: str):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    df.to_csv(path, index=False)
    print(f"Saved {filename} to {folder}")
    return path
  