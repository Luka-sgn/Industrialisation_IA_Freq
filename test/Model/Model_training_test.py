import pytest
from sklearn.ensemble import RandomForestClassifier
from src.Model.model_training import train_final_model
import pandas as pd

def test_train_final_model():
    # Données factices
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    X_val = pd.DataFrame({'feature1': [1, 2], 'feature2': [4, 5]})
    y_val = pd.Series([0, 1])

    # Paramètres fictifs
    best_params = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "min_child_weight": 1,
        "scale_pos_weight": 1,
        "random_state": 42,
        "eval_metric": ["aucpr", "auc"]
    }

    # Appel de la fonction d'entraînement
    model = train_final_model(X_train, y_train, best_params, X_val, y_val)

    # Vérifier que le modèle a été entraîné sans erreur
    assert model is not None
    assert hasattr(model, 'predict')  # Vérifier que le modèle a bien un attribut 'predict'
