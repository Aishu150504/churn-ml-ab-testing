import pandas as pd

def data_drift_check(train_df, new_df):
    """
    Simple data drift check using mean difference
    """
    drift = (train_df.mean() - new_df.mean()).abs()
    return drift
