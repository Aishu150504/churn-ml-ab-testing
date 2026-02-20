import pandas as pd

def load_data(path):
    """
    Loads raw churn dataset
    """
    df = pd.read_csv(path)
    print(f"Data Loaded: {df.shape}")
    return df
