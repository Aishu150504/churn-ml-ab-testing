import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            solver="lbfgs"
        ))
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, "models/churn_model.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

    return X_test, y_test
