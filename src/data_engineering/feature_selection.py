def select_features(df_cat, df_ord, df_con, X_train, X_test):
    selected_vars = (
            df_cat["Variable_Cat"].tolist() +
            df_ord["Variable_Ord"].tolist() +
            df_con["Variable_Con"].tolist()
    )

    cols_to_keep = ['ID', 'ANNEE_ASSURANCE']

    selected_vars = cols_to_keep + [col for col in selected_vars if col not in cols_to_keep]

    X_train_filtered = X_train[selected_vars]
    X_test_filtered = X_test[selected_vars]

    X_train_filtered.to_csv("Data/X_train_filtered_freq.csv", index=False)
    X_test_filtered.to_csv("Data/X_test_filtered_freq.csv", index=False)

    return X_train_filtered, X_test_filtered
