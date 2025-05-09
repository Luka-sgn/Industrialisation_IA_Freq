import pandas as pd

def preprocess_y_train(y_train, X_train):
    y_train["NB_SINISTRES"] = y_train["FREQ"] * X_train["ANNEE_ASSURANCE"]
    y_train["SINISTRE"] = (y_train["NB_SINISTRES"] >= 1).astype(int)
    y_train.to_csv("Y_train_sinistre_2classes.csv", index=False)

    y_train["SINISTRE"] = 0
    y_train.loc[(y_train["NB_SINISTRES"] >= 1) & (y_train["NB_SINISTRES"] <= 2), "SINISTRE"] = 1
    y_train.loc[y_train["NB_SINISTRES"] > 2, "SINISTRE"] = 2

    y_train.to_csv("Y_train_sinistre_3classes.csv", index=False)

    return y_train
