def predict_expected_claims(model, X, threshold, mean_1plus):
    """Prédit l'espérance de sinistres en combinant classification et espérance conditionnelle."""
    proba_1plus = model.predict_proba(X)[:, 1]
    predictions_1plus = (proba_1plus >= threshold).astype(int)
    return predictions_1plus * mean_1plus  # E[N] = P(1+) * E[N|1+]
