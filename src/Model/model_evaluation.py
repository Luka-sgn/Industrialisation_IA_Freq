from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

def evaluate_model(final_model, X_val, y_val, best_threshold):
    y_pred_proba = final_model.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    print("\n📊 Performance finale:")
    print(classification_report(y_val, y_pred, target_names=["0 sinistre", "1+ sinistres"]))
    print("\nMatrice de confusion:")
    print(confusion_matrix(y_val, y_pred))

    return best_threshold, y_pred
