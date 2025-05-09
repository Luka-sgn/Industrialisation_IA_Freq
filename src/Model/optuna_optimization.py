import optuna
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

def optimize_model(X_tr, y_tr, X_val, y_val, scale_pos_weight):
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "min_child_weight": trial.suggest_int("min_child", 1, 10),
            "scale_pos_weight": trial.suggest_float("pos_weight", scale_pos_weight*0.5, scale_pos_weight*1.5),
            "random_state": 42,
            "eval_metric": ["aucpr", "auc"]
        }

        model = XGBClassifier(**params)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        proba = model.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, proba)
        ap = average_precision_score(y_val, proba)  # AUC-PR (meilleur pour déséquilibre)

        return 0.7 * ap + 0.3 * auc  # Poids plus fort sur AUC-PR

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, timeout=3600)

    # Récupérer les meilleurs paramètres
    best_params = study.best_params
    best_value = study.best_value

    # Sauvegarder les meilleurs paramètres dans un fichier pickle (.pkl)
    with open('params/best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)

    return best_value, best_params