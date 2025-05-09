!pip install optuna
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import  XGBClassifier
from sklearn.metrics import mean_squared_error
import re
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
import optuna
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.utils import class_weight
import pandas as pd

# Chargement des fichiers
X_train = pd.read_csv("/Users/lukasegouin/Documents/CYU/ING3/data_science/processed_data/Xtrain/X_train_filtered_freq.csv")
X_test = pd.read_csv("/Users/lukasegouin/Documents/CYU/ING3/data_science/processed_data/Xtest/X_test_filtered_freq.csv")
Y_train = pd.read_csv("/Users/lukasegouin/Documents/CYU/ING3/data_science/processed_data/Xtest/Y_train_sinistre_2classes.csv")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", Y_train.shape)

from sklearn.metrics import roc_auc_score
# Extraction des cibles
y_train = Y_train["SINISTRE"]
y_train_full = Y_train  # pour accéder à NB_SINISTRES

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)

# ---------------------------------------------
# 🎯 Configuration initiale
# ---------------------------------------------
y_train = Y_train["SINISTRE"]  # 0 vs 1
y_train_full = Y_train  # Contient NB_SINISTRES

print("Shapes initiaux:")
print("X_train:", X_train.shape, "| y_train:", y_train.shape)
print("\nDistribution des classes (0 vs 1+):")
print(y_train.value_counts(normalize=True))

# Calcul de l'espérance conditionnelle pour 1+
mean_1plus = y_train_full.loc[y_train == 1, "NB_SINISTRES"].mean()
print(f"\nEspérance conditionnelle (1+): {mean_1plus:.2f} sinistres")

# ---------------------------------------------
# 🛠️ Préparation des données
# ---------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train  # Préservation du ratio de classes
)

# Calcul des poids de classe (
# Poids
scale_pos_weight = len(y_tr[y_tr==0]) / len(y_tr[y_tr==1])  # Ratio 0/1+

# ---------------------------------------------
# 🎯 Fonction d'optimisation Optuna
# ---------------------------------------------
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

    # Entraînemen
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Prédictions probabilistes
    proba = model.predict_proba(X_val)[:, 1]

    # Métriques principales
    auc = roc_auc_score(y_val, proba)
    ap = average_precision_score(y_val, proba)  # AUC-PR (meilleur pour déséquilibre)

    # On maximise une combinaison des deux métriques
    return 0.7 * ap + 0.3 * auc  # Poids plus fort sur AUC-PR

# 🔍 Lancement Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, timeout=3600)

print("\n🔍 Résultats optimisation:")
print("Best score (combinaison AUC-PR + AUC):", study.best_value)
print("Meilleurs paramètres:", study.best_params)

# ---------------------------------------------
# 🚀 Entraînement final avec meilleurs paramètres
# ---------------------------------------------
best_params = study.best_params.copy()
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

# ---------------------------------------------
# 📊 Évaluation complète
# ---------------------------------------------
from sklearn.metrics import classification_report, confusion_matrix

y_pred_proba = final_model.predict_proba(X_val)[:, 1]

# Optimisation du seuil pour F1-score
precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_threshold = thresholds[np.argmax(f1_scores)]
y_pred = (y_pred_proba >= best_threshold).astype(int)

print("\n📊 Performance finale:")
print(classification_report(y_val, y_pred, target_names=["0 sinistre", "1+ sinistres"]))
print("\nMatrice de confusion:")
print(confusion_matrix(y_val, y_pred))

# ---------------------------------------------
# 💡 Application actuarielle
# ---------------------------------------------
def predict_expected_claims(model, X, threshold=best_threshold):
    """Prédit l'espérance de sinistres en combinant classification et espérance conditionnelle."""
    proba_1plus = model.predict_proba(X)[:, 1]
    predictions_1plus = (proba_1plus >= threshold).astype(int)
    return predictions_1plus * mean_1plus  # E[N] = P(1+) * E[N|1+]


expected_claims = predict_expected_claims(final_model, X_val)
print("\nEspérance prédite pour les 5 premiers cas (val):")
print(expected_claims[:5])
print("Valeurs réelles (y_val):")
print(y_val.head().values)

# ---------------------------------------------
# 💾 Injection des prédictions + Sauvegarde
# ---------------------------------------------

# 🎯 On injecte les prédictions dans un dataframe
results = X_test[["ID", "ANNEE_ASSURANCE"]].copy()

# Prédiction finale sur l'ensemble test filtré
X_test_filtered = X_test.loc[X_test.index]
pred_freq_test = predict_expected_claims(final_model, X_test_filtered)

# Injection dans les lignes concernées
results.loc[X_test_filtered.index, "FREQ_prediction"] = pred_freq_test

results.to_csv("/content/freq.csv", index=False)
print("\n✅ Fichier 'freq.csv' sauvegardé avec les prédictions d'espérance de fréquence.")
