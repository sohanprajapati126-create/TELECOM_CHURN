import unittest
import joblib
from sklearn.linear_model import LogisticRegression

class TestModelTraining(unittest.TestCase):
    def test_model_training(self):
        # Load the saved model
        model = joblib.load('model/Telecome_model.pkl')
        
        # Check that it is a LogisticRegression model
        self.assertIsInstance(model, LogisticRegression)
        
        # Check that it has at least 4 features (coefficients)
        self.assertGreaterEqual(len(model.coef_[0]), 4)

if _name_ == '_main_':
    unittest.main()
