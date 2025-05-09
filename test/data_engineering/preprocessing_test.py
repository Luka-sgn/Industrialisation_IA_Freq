import unittest
import pandas as pd
from src.data_engineering.preprocessing import preprocess_y_train

class TestPreprocessYTrain(unittest.TestCase):

    def test_preprocess_y_train(self):
        # Exemple de données
        X_train = pd.DataFrame({'ANNEE_ASSURANCE': [1, 2, 3]})
        y_train = pd.DataFrame({'FREQ': [0, 1, 0], 'SINISTRE': [0, 1, 0]})

        # Appel de la fonction
        processed_y_train = preprocess_y_train(y_train, X_train)

        # Vérifier que la transformation a bien eu lieu
        self.assertIn('NB_SINISTRES', processed_y_train.columns)  # Vérifier si la colonne 'NB_SINISTRES' existe
        self.assertEqual(processed_y_train['SINISTRE'].dtype, 'int64')  # Vérifier que le type de la colonne 'SINISTRE' est 'int64'

if __name__ == '__main__':
    unittest.main()
