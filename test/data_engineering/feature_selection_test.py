from src.data_engineering.feature_selection import select_features
import pandas as pd

def test_select_features():
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
    assert len(selected_features) > 0
