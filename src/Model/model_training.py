from xgboost import XGBClassifier

def train_final_model(X_train, y_train, best_params, X_val, y_val):
    best_params.update({
        "objective": "binary:logistic",
        "random_state": 42,
        "eval_metric": ["aucpr", "auc"]
    })

    final_model = XGBClassifier(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    return final_model
