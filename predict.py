import joblib
import pandas as pd
import numpy as np

def predict(input_dict):
    """
    input_dict: dict of feature_name -> value
    """

    # Load trained pipeline and schema
    model = joblib.load("models/churn_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")

    # Create DataFrame with ALL expected features
    X = pd.DataFrame(columns=feature_names)
    X.loc[0] = 0  # default fill

    # Update with provided inputs
    for key, value in input_dict.items():
        if key in X.columns:
            X.at[0, key] = value

    # Ensure numeric and no NaNs
    X = X.astype(float)
    X = X.fillna(0)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }
