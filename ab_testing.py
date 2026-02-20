import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from src.predict import predict

def ab_test(n_users=1000):
    """
    Simulate A/B testing for churn reduction strategy
    Control: No intervention
    Treatment: Retention offer applied
    """

    np.random.seed(42)

    results = []

    for i in range(n_users):
        # Simulated customer features (simplified)
        customer = {
            "tenure": np.random.randint(1, 60),
            "MonthlyCharges": np.random.randint(20, 120),
            "TotalCharges": np.random.randint(100, 5000),
            "gender_Male": np.random.randint(0, 2),
            "SeniorCitizen": np.random.randint(0, 2),
            "Partner_Yes": np.random.randint(0, 2),
            "Dependents_Yes": np.random.randint(0, 2),
            "PhoneService_Yes": 1,
            "MultipleLines_Yes": np.random.randint(0, 2),
            "InternetService_Fiber optic": np.random.randint(0, 2),
            "OnlineSecurity_Yes": np.random.randint(0, 2),
            "OnlineBackup_Yes": np.random.randint(0, 2),
            "DeviceProtection_Yes": np.random.randint(0, 2),
            "TechSupport_Yes": np.random.randint(0, 2),
            "StreamingTV_Yes": np.random.randint(0, 2),
            "StreamingMovies_Yes": np.random.randint(0, 2),
            "Contract_Two year": np.random.randint(0, 2),
            "PaperlessBilling_Yes": np.random.randint(0, 2),
            "PaymentMethod_Electronic check": np.random.randint(0, 2),
        }

        group = np.random.choice(["control", "treatment"])

        pred = predict(customer)["churn_prediction"]

        # Simulate treatment effect (treatment reduces churn by 10%)
        if group == "treatment" and pred == 1:
            pred = np.random.choice([0, 1], p=[0.1, 0.9])

        results.append([group, pred])

    df = pd.DataFrame(results, columns=["group", "churn"])
    import os
    os.makedirs("logs", exist_ok=True)

    df.to_csv("logs/ab_test_results.csv", index=False)
    print("A/B test results saved to logs/ab_test_results.csv")


    contingency = pd.crosstab(df["group"], df["churn"])

    chi2, p_value, _, _ = chi2_contingency(contingency)

    print("\nA/B TEST RESULTS")
    print(contingency)
    print(f"\nChi-square p-value: {round(p_value, 4)}")

    if p_value < 0.05:
        print("✅ Statistically significant difference detected")
    else:
        print("❌ No statistically significant difference")

    return contingency, p_value

