import unittest
import numpy as np

from src.Model.model_evaluation import evaluate_model


# Mock d’un modèle
class MockModel:
    def __init__(self, proba_outputs):
        self.proba_outputs = proba_outputs

    def predict_proba(self, X):
        return self.proba_outputs

# Classe de test
class TestEvaluateModel(unittest.TestCase):

    def setUp(self):
        # Simule une sortie de predict_proba (probas pour 0 et 1)
        self.mock_model = MockModel(
            np.array([[0.9, 0.1], [0.3, 0.7], [0.4, 0.6], [0.8, 0.2]])
        )
        self.X_val = np.zeros((4, 2))  # peu importe les valeurs
        self.y_val = np.array([0, 1, 1, 0])

    def test_output_format_and_values(self):
        threshold, y_pred = evaluate_model(self.mock_model, self.X_val, self.y_val)

        # Test du type et contenu
        self.assertIsInstance(threshold, float)
        self.assertTrue(0.0 <= threshold <= 1.0)

        self.assertIsInstance(y_pred, np.ndarray)
        self.assertTrue(np.all(np.isin(y_pred, [0, 1])))

        # Test que la longueur est correcte
        self.assertEqual(len(y_pred), len(self.y_val))

if __name__ == '__main__':
    unittest.main()
