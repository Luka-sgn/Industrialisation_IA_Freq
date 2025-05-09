import pandas as pd

def load_data(X_train_path, X_test_path, Y_train_path):
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    Y_train = pd.read_csv(Y_train_path)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", Y_train.shape)

    y_train = Y_train["SINISTRE"]
    y_train_full = Y_train  # Pour accéder à NB_SINISTRES

    return X_train, X_test, y_train, y_train_full
