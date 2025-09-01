import unittest
import joblib
import pandas as pd

class TestModel(unittest.TestCase):
    def setUp(self):
        # Load dataset
        self.df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        self.X = self.df.drop(['customerID', 'Churn'], axis=1)
        self.y = self.df['Churn']
        
        # Load trained pipeline (or model)
        self.model = joblib.load("model/Telecome_model.pkl")

    def test_model_predict(self):
        # Take first 5 samples
        X_sample = self.X.head(5)
        preds = self.model.predict(X_sample)
        
        # Check predictions length matches inputs
        self.assertEqual(len(preds), len(X_sample))

if __name__ == "__main__":
    unittest.main()
