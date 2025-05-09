import pandas as pd

def load_data(input_train_path, input_y_train_path, input_test_path):
    print("Chargement des données...")
    X_train = pd.read_csv(input_train_path)
    y_train = pd.read_csv(input_y_train_path)
    X_test = pd.read_csv(input_test_path)
    print("Données chargées avec succès.")

    return X_train, y_train, X_test
