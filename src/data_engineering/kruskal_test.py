import pandas as pd
from scipy.stats import kruskal

def calculate_kruskal_for_continuous_columns(X_train, y_train, numerical_columns):
    results_kruskal = []
    for col in numerical_columns:
        try:
            g0 = X_train[y_train["SINISTRE"] == 0][col].dropna()
            g1 = X_train[y_train["SINISTRE"] == 1][col].dropna()
            stat, pval = kruskal(g0, g1)
            results_kruskal.append((col, stat, pval))
        except:
            continue
    return pd.DataFrame(results_kruskal, columns=["Variable_Con", "H_stat", "p_value"]).sort_values(by="p_value")
