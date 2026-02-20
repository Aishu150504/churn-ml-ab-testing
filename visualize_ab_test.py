import pandas as pd
import matplotlib.pyplot as plt

# Load A/B test results
df = pd.read_csv("logs/ab_test_results.csv")

# Convert churn to conversion (retention)
# churn = 1 → left
# churn = 0 → stayed (success)
df["conversion"] = 1 - df["churn"]

# Compute conversion (retention) rate
conversion_rates = df.groupby("group")["conversion"].mean()

print("Retention (Conversion) Rates:")
print(conversion_rates)

# Plot
conversion_rates.plot(kind="bar")
plt.title("A/B Test Retention Rate Comparison")
plt.ylabel("Retention Rate")
plt.xlabel("Group")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.show()
