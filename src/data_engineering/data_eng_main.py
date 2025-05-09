import pandas as pd
from src.data_engineering.data_loading import load_data
from src.data_engineering.preprocessing import preprocess_y_train
from src.data_engineering.cramer_v import calculate_cramers_v_for_categorical_columns
from src.data_engineering.kruskal_test import calculate_kruskal_for_continuous_columns
from src.data_engineering.feature_selection import select_features
from src.data_engineering.utils import print_class_distribution

def main():
    # Chargement des données
    X_train, y_train, X_test = load_data("/path/to/train_input_cleaned.csv", "/path/to/Ytrain.csv", "/path/to/test_input_cleaned.csv")

    # Prétraitement des données
    y_train = preprocess_y_train(y_train, X_train)

    # Affichage de la distribution des classes
    print_class_distribution(y_train)

    # Cramér's V pour les variables catégorielles
    categorical_columns = ['ACTIVIT2', 'VOCATION', 'ADOSS', ...]  # Continuez avec la liste complète
    df_cat = calculate_cramers_v_for_categorical_columns(X_train, y_train, categorical_columns)

    # Test de Kruskal-Wallis pour les variables continues
    numerical_columns = ['ANCIENNETE', 'CARACT2', 'DUREE_REQANEUF', ...]  # Continuez avec la liste complète
    df_con = calculate_kruskal_for_continuous_columns(X_train, y_train, numerical_columns)

    # Sélection des caractéristiques
    X_train_filtered, X_test_filtered = select_features(df_cat, df_ord, df_con, X_train, X_test)

    print("X_train shape:", X_train_filtered.shape)
    print("X_test shape:", X_test_filtered.shape)

if __name__ == "__main__":
    main()
