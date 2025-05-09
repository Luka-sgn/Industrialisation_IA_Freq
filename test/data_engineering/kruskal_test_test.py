import pandas as pd
from src.data_engineering.kruskal_test import calculate_kruskal_for_continuous_columns

def test_kruskal_test():
    # Exemple de données
    X_train = {'col1': [1, 2, 3, 4, 5], 'col2': [6, 7, 8, 9, 10]}
    y_train = {'SINISTRE': [0, 1, 0, 1, 0]}

    # Appel de la fonction
    result = calculate_kruskal_for_continuous_columns(X_train, y_train, ['col1', 'col2'])

    # Vérifier que la sortie est correcte
    assert isinstance(result, pd.DataFrame)
    assert 'H_stat' in result.columns
    assert 'p_value' in result.columns
