import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Use relative path (works on GitHub Actions/Linux and locally if file is in repo)
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocess the dataset
X = pd.get_dummies(df.drop(['customerID', 'Churn'], axis=1), drop_first=True)
y = df['Churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=101
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Ensure model folder exists
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(model, "model/Telecom_model.pkl")

