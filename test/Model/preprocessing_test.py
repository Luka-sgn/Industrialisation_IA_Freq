import unittest
from unittest.mock import patch
import pandas as pd
from src.Model.preprocessing import prepare_data

class TestPreprocessYTrain(unittest.TestCase):

    @patch('pandas.DataFrame.to_csv')  # Mocker la méthode to_csv
    def test_preprocess_y_train(self, mock_to_csv):
        # Exemple de données
        X_train = pd.DataFrame({'ANNEE_ASSURANCE': [1, 2, 3]})
        y_train = pd.DataFrame({'FREQ': [0, 1, 0], 'SINISTRE': [0, 1, 0]})

        # Appel de la fonction
        processed_y_train = prepare_data(y_train, X_train)

        # Vérifier que la transformation a bien eu lieu
        self.assertIn('NB_SINISTRES', processed_y_train.columns, "La colonne 'NB_SINISTRES' n'a pas été ajoutée.")
        self.assertEqual(processed_y_train['SINISTRE'].dtype, 'int64', "Le type de la colonne 'SINISTRE' n'est pas 'int64'.")

        # Vérifier que la méthode to_csv n'a pas été appelée
        mock_to_csv.assert_not_called()

if __name__ == '__main__':
    unittest.main()
