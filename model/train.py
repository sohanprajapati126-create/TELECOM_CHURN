import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocess with get_dummies
X = pd.get_dummies(df.drop(['customerID', 'Churn'], axis=1), drop_first=True)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Save model + feature columns
joblib.dump(model, "model/Telecome_model.pkl")
joblib.dump(X.columns, "model/Telecome_columns.pkl")
print("Model and columns saved successfully.")
