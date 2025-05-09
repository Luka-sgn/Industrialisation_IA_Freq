from src.Model.preprocessing import prepare_data
import pandas as pd

def test_preprocess_y_train():
    # Exemple de données
    X_train = pd.DataFrame({'ANNEE_ASSURANCE': [1, 2, 3]})
    y_train = pd.DataFrame({'FREQ': [0, 1, 0], 'SINISTRE': [0, 1, 0]})

    # Appel de la fonction
    processed_y_train = prepare_data(y_train, X_train)

    # Vérifier que la transformation a bien eu lieu
    assert 'NB_SINISTRES' in processed_y_train.columns
    assert processed_y_train['SINISTRE'].dtype == 'int64'
