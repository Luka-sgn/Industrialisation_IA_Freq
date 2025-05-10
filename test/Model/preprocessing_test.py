import unittest
import pandas as pd
import numpy as np

from src.Model.preprocessing import prepare_data


# Classe de test
class TestPrepareData(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame({
            "feat1": np.random.rand(100),
            "feat2": np.random.rand(100)
        })
        # 70 zéros, 30 uns
        self.y = pd.Series([0]*70 + [1]*30)

    def test_split_shapes(self):
        X_tr, X_val, y_tr, y_val, sw = prepare_data(self.X, self.y)
        self.assertEqual(len(X_tr), 80)
        self.assertEqual(len(X_val), 20)
        self.assertEqual(len(y_tr), 80)
        self.assertEqual(len(y_val), 20)

    def test_stratification_preserved(self):
        _, _, y_tr, y_val, _ = prepare_data(self.X, self.y)
        # approx 56/24 for 70/30 split
        self.assertAlmostEqual(y_tr.mean(), 0.3, delta=0.05)
        self.assertAlmostEqual(y_val.mean(), 0.3, delta=0.05)

    def test_scale_pos_weight(self):
        _, _, y_tr, _, sw = prepare_data(self.X, self.y)
        expected_ratio = len(y_tr[y_tr==0]) / len(y_tr[y_tr==1])
        self.assertAlmostEqual(sw, expected_ratio)

if __name__ == '__main__':
    unittest.main()
