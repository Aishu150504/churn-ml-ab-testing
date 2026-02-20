import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8")
sns.set_palette("Set2")

df = pd.read_csv("data/raw/telco_churn.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# Missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(df.isnull().sum())

# Churn distribution
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

# Tenure vs Churn
sns.boxplot(x="Churn", y="tenure", data=df)
plt.show()

# Monthly Charges vs Churn
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.show()

# Contract vs Churn
sns.countplot(x="Contract", hue="Churn", data=df)
plt.xticks(rotation=30)
plt.show()

# Correlation
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True)
plt.show()

df.to_csv("data/processed/clean_data.csv", index=False)
