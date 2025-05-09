import unittest
import pandas as pd
from src.data_engineering.cramer_v import cramers_v

class TestCramersV(unittest.TestCase):

    def test_cramers_v(self):
        # Données d'exemple
        data = {'col1': ['A', 'B', 'A', 'B', 'A'], 'col2': ['X', 'X', 'Y', 'Y', 'X']}
        df = pd.DataFrame(data)

        # Calcul de Cramér's V
        result = cramers_v(df['col1'], df['col2'])

        # Vérifier que le résultat est un nombre
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)  # Cramér's V doit être >= 0
        self.assertLessEqual(result, 1)    # Cramér's V doit être <= 1

if __name__ == '__main__':
    unittest.main()
