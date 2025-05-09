import pandas as pd

def preprocess_y_train(y_train, X_train):
    y_train["NB_SINISTRES"] = y_train["FREQ"] * X_train["ANNEE_ASSURANCE"]
    y_train["SINISTRE"] = (y_train["NB_SINISTRES"] >= 1).astype(int)
    y_train.to_csv("Classes/Y_train_sinistre_2classes.csv", index=False)

    return y_train
