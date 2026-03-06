import pandas as pd
from pathlib import Path

def load_data(file_path: Path,columns: list[str],sep: str = ",",encoding: str = "utf-8", print_head: bool = False) -> pd.DataFrame:
    
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, sep=sep, names=columns, engine='python', encoding=encoding)
    
    return df

