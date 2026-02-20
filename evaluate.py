from sklearn.metrics import accuracy_score, roc_auc_score
import json
import joblib

def evaluate(X_test, y_test):
    model = joblib.load("models/churn_model.pkl")

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4)
    }

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation Metrics:", metrics)
    return metrics
