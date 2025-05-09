from src.Model.data_loading import load_data
from src.Model.preprocessing import prepare_data
from src.Model.optuna_optimization import optimize_model
from src.Model.model_training import train_final_model
from src.Model.model_evaluation import evaluate_model
from src.Model.utils import predict_expected_claims

def main():
    # Chargement des données
    X_train, X_test, y_train, y_train_full = load_data(
        "/Users/lukasegouin/IdeaProjects/Industrialisation_IA_Freq/src/data_engineering/Data/X_train_filtered_freq.csv",
        "/Users/lukasegouin/IdeaProjects/Industrialisation_IA_Freq/src/data_engineering/Data/X_test_filtered_freq.csv",
        "/Users/lukasegouin/IdeaProjects/Industrialisation_IA_Freq/src/data_engineering/Classes/Y_train_sinistre_2classes.csv"
    )

    # Préparation des données
    X_tr, X_val, y_tr, y_val, scale_pos_weight = prepare_data(X_train, y_train)

    # Optimisation avec Optuna
    best_value, best_params = optimize_model(X_tr, y_tr, X_val, y_val, scale_pos_weight)

    print("\n🔍 Résultats optimisation:")
    print("Best score (combinaison AUC-PR + AUC):", best_value)
    print("Meilleurs paramètres:", best_params)

    # Entraînement final avec les meilleurs paramètres
    final_model = train_final_model(X_train, y_train, best_params, X_val, y_val)

    # Évaluation du modèle
    best_threshold, y_pred = evaluate_model(final_model, X_val, y_val)

    # Application actuarielle
    mean_1plus = y_train_full.loc[y_train == 1, "NB_SINISTRES"].mean()
    expected_claims = predict_expected_claims(final_model, X_val, best_threshold, mean_1plus)

    # Sauvegarde des résultats
    results = X_test[["ID", "ANNEE_ASSURANCE"]].copy()
    X_test_filtered = X_test.loc[X_test.index]
    pred_freq_test = predict_expected_claims(final_model, X_test_filtered, best_threshold, mean_1plus)
    results.loc[X_test_filtered.index, "FREQ_prediction"] = pred_freq_test

    results.to_csv("/content/freq.csv", index=False)
    print("\n✅ Fichier 'freq.csv' sauvegardé avec les prédictions d'espérance de fréquence.")

if __name__ == "__main__":
    main()
