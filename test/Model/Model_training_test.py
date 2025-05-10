import unittest
import numpy as np
from xgboost import XGBClassifier
from sklearn.datasets import make_classification

from src.Model.model_training import train_final_model


# Classe de test
class TestTrainFinalModel(unittest.TestCase):

    def setUp(self):
        self.X_train, self.y_train = make_classification(
            n_samples=100, n_features=10, n_informative=5, random_state=42
        )
        self.X_val, self.y_val = make_classification(
            n_samples=20, n_features=10, n_informative=5, random_state=123
        )
        self.best_params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1
        }

    def test_model_training(self):
        model = train_final_model(
            self.X_train, self.y_train,
            best_params=self.best_params.copy(),
            X_val=self.X_val, y_val=self.y_val
        )

        self.assertIsInstance(model, XGBClassifier)
        self.assertTrue(hasattr(model, "feature_importances_"))

        # Vérifie que les paramètres sont bien pris en compte
        self.assertEqual(model.get_params()["n_estimators"], 10)
        self.assertEqual(model.get_params()["max_depth"], 3)
        self.assertEqual(model.get_params()["learning_rate"], 0.1)
        self.assertEqual(model.get_params()["objective"], "binary:logistic")

if __name__ == '__main__':
    unittest.main()
