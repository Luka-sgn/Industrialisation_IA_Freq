import unittest
import pandas as pd
import os
from src.data_engineering.feature_selection import select_features

class TestSelectFeatures(unittest.TestCase):

    def setUp(self):
        # ✅ S'assurer que le dossier Data existe
        self.output_dir = "Data"
        os.makedirs(self.output_dir, exist_ok=True)

        self.df_cat = pd.DataFrame({"Variable_Cat": ["CAT1", "CAT2"]})
        self.df_ord = pd.DataFrame({"Variable_Ord": ["ORD1"]})
        self.df_con = pd.DataFrame({"Variable_Con": ["CON1"]})

        columns = ['ID', 'ANNEE_ASSURANCE', 'CAT1', 'CAT2', 'ORD1', 'CON1', 'OTHER']
        data = [[1, 5, 'a', 'b', 3, 1.5, 'z'], [2, 3, 'c', 'd', 2, 2.0, 'y']]

        self.X_train = pd.DataFrame(data, columns=columns)
        self.X_test = pd.DataFrame(data, columns=columns)

        # Chemins attendus pour les fichiers CSV
        self.train_csv = os.path.join(self.output_dir, "X_train_filtered_freq.csv")
        self.test_csv = os.path.join(self.output_dir, "X_test_filtered_freq.csv")

    def tearDown(self):
        # 🔁 Supprime les fichiers générés après chaque test
        for f in [self.train_csv, self.test_csv]:
            if os.path.exists(f):
                os.remove(f)

    def test_column_selection_and_order(self):
        X_train_f, X_test_f = select_features(
            self.df_cat, self.df_ord, self.df_con, self.X_train, self.X_test
        )
        expected_cols = ['ID', 'ANNEE_ASSURANCE', 'CAT1', 'CAT2', 'ORD1', 'CON1']
        self.assertEqual(X_train_f.columns.tolist(), expected_cols)
        self.assertEqual(X_test_f.columns.tolist(), expected_cols)

    def test_csv_output_exists(self):
        select_features(
            self.df_cat, self.df_ord, self.df_con, self.X_train, self.X_test
        )
        self.assertTrue(os.path.exists(self.train_csv))
        self.assertTrue(os.path.exists(self.test_csv))

if __name__ == '__main__':
    unittest.main()
