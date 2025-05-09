import unittest
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from src.Model.model_evaluation import evaluate_model

class TestEvaluateModel(unittest.TestCase):

    def test_evaluate_model(self):
        # Données factices
        X_test = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y_test = pd.Series([0, 1, 0])

        # Modèle factice
        model = RandomForestClassifier()
        model.fit(X_test, y_test)

        # Appel de la fonction d'évaluation
        best_threshold, y_pred = evaluate_model(model, X_test, y_test, 0.5)

        # Vérifier que le seuil et les prédictions sont retournés
        self.assertIsNotNone(best_threshold, "Le seuil retourné est None.")
        self.assertEqual(len(y_pred), len(X_test), "La longueur des prédictions ne correspond pas à la longueur des données d'entrée.")

if __name__ == '__main__':
    unittest.main()
