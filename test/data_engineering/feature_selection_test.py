import unittest
import pandas as pd
from src.data_engineering.feature_selection import select_features

class TestFeatureSelection(unittest.TestCase):

    def test_select_features(self):
        # Exemple de données
        df_cat = pd.DataFrame({'cat1': [1, 2, 3], 'cat2': [4, 5, 6]})
        df_ord = pd.DataFrame({'ord1': [7, 8, 9], 'ord2': [10, 11, 12]})
        df_con = pd.DataFrame({'con1': [13, 14, 15], 'con2': [16, 17, 18]})

        # Exemple de données de test
        X_train = pd.DataFrame({'cat1': [1, 2], 'cat2': [4, 5], 'ord1': [7, 8], 'ord2': [10, 11], 'con1': [13, 14]})
        X_test = pd.DataFrame({'cat1': [1], 'cat2': [4], 'ord1': [7], 'ord2': [10], 'con1': [13]})

        # Appel de la fonction
        selected_features = select_features(df_cat, df_ord, df_con, X_train, X_test)

        # Vérifier que la sélection de features retourne les bonnes variables
        self.assertGreater(len(selected_features), 0, "La sélection des features n'a pas retourné de caractéristiques.")
        self.assertTrue(all(col in X_train.columns for col in selected_features), "Certaines caractéristiques sélectionnées ne sont pas présentes dans X_train.")

if __name__ == '__main__':
    unittest.main()