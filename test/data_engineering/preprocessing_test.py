import unittest
import pandas as pd
import os
from src.data_engineering.preprocessing import preprocess_y_train

class TestPreprocessYTrain(unittest.TestCase):

    def setUp(self):
        self.test_dir = os.path.join("Classes_test")
        os.makedirs(self.test_dir, exist_ok=True)

        self.y_train = pd.DataFrame({
            "FREQ": [0.0, 0.1, 0.5, 1.0]
        })
        self.X_train = pd.DataFrame({
            "ANNEE_ASSURANCE": [1, 5, 2, 1]
        })

    def test_nb_sinistres_calculation(self):
        path = os.path.join(self.test_dir, "Y_train_sinistre_2classes_test.csv")
        result = preprocess_y_train(self.y_train.copy(), self.X_train, path)
        expected_nb_sinistres = [0.0, 0.5, 1.0, 1.0]
        self.assertListEqual(result["NB_SINISTRES"].tolist(), expected_nb_sinistres)

    def test_sinistre_classification(self):
        path = os.path.join(self.test_dir, "Y_train_sinistre_2classes_test.csv")
        result = preprocess_y_train(self.y_train.copy(), self.X_train, path)
        expected_sinistre = [0, 0, 1, 1]
        self.assertListEqual(result["SINISTRE"].tolist(), expected_sinistre)

    def test_csv_output(self):
        path = os.path.join(self.test_dir, "Y_train_sinistre_2classes_test.csv")
        preprocess_y_train(self.y_train.copy(), self.X_train, path)
        self.assertTrue(os.path.exists(path))

if __name__ == '__main__':
    unittest.main()
