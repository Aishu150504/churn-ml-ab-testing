from sklearn.preprocessing import LabelEncoder

def engineer_features(df):
    """
    Encodes categorical features for churn model
    """

    # Drop customer ID if present
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    encoder = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = encoder.fit_transform(df[col])

    return df
