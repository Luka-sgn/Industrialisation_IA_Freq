from sklearn.model_selection import train_test_split

def prepare_data(X_train, y_train):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    scale_pos_weight = len(y_tr[y_tr==0]) / len(y_tr[y_tr==1])  # Ratio 0/1+

    return X_tr, X_val, y_tr, y_val, scale_pos_weight
