import unittest
import pandas as pd
from src.data_engineering.kruskal_test import calculate_kruskal_for_continuous_columns

class TestKruskalTest(unittest.TestCase):

    def test_kruskal_test(self):
        # Exemple de données
        X_train = {'col1': [1, 2, 3, 4, 5], 'col2': [6, 7, 8, 9, 10]}
        y_train = {'SINISTRE': [0, 1, 0, 1, 0]}

        # Convertir X_train et y_train en DataFrame
        X_train_df = pd.DataFrame(X_train)
        y_train_df = pd.DataFrame(y_train)

        # Appel de la fonction
        result = calculate_kruskal_for_continuous_columns(X_train_df, y_train_df, ['col1', 'col2'])

        # Vérifier que la sortie est correcte
        self.assertIsInstance(result, pd.DataFrame)  # Vérifie que le résultat est un DataFrame
        self.assertIn('H_stat', result.columns)  # Vérifie que 'H_stat' est une colonne du DataFrame
        self.assertIn('p_value', result.columns)  # Vérifie que 'p_value' est une colonne du DataFrame

if __name__ == '__main__':
    unittest.main()
