import pandas as pd
from src.data_drift import data_drift_check

# Example training data
train_df = pd.DataFrame({
    "tenure": [1, 2, 3, 4, 5],
    "MonthlyCharges": [50, 55, 60, 65, 70]
})

# Example new data
new_df = pd.DataFrame({
    "tenure": [2, 3, 4, 5, 6],
    "MonthlyCharges": [52, 58, 63, 67, 75]
})

drift = data_drift_check(train_df, new_df)

print("📊 Data Drift (Mean Difference):")
print(drift)

