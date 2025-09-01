import os
import unittest
import joblib

class TestModel(unittest.TestCase):

    def test_model_file_exists(self):
        """Check if the trained model file exists"""
        self.assertTrue(
            os.path.exists("model/Churn_model.pkl"),
            "❌ Model file not found. Run train.py before testing."
        )

    def test_model_load(self):
        """Try loading the model if it exists"""
        if os.path.exists("model/Churn_model.pkl"):
            model = joblib.load("model/Churn_model.pkl")
            self.assertIsNotNone(model, "❌ Model could not be loaded.")
        else:
            self.skipTest("Model file not found. Skipping load test.")

if __name__ == "__main__":
    unittest.main()
