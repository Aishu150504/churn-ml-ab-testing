import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def preprocess():
    """
    Load and clean raw churn data
    """

    df = pd.read_csv("data/raw/telco_churn.csv")

    # Convert TotalCharges to numeric (fixes blank values)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop customerID (not useful)
    df.drop("customerID", axis=1, inplace=True)

    # Separate numeric and categorical columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # Impute numeric columns with median
    num_imputer = SimpleImputer(strategy="median")
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Impute categorical columns with most frequent
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df
