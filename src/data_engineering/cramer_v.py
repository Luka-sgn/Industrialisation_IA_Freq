import pandas as pd
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table)[0]
    n = table.sum().sum()
    r, k = table.shape
    phi2 = chi2 / n
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    return (phi2_corr / min(k_corr-1, r_corr-1))**0.5

def calculate_cramers_v_for_categorical_columns(X_train, y_train, categorical_columns):
    results_cramer_cat = []
    for col in categorical_columns:
        try:
            vcat = cramers_v(X_train[col], y_train["SINISTRE"])
            results_cramer_cat.append((col, vcat))
        except:
            continue
    return pd.DataFrame(results_cramer_cat, columns=["Variable_Cat", "Cramers_V"]).sort_values(by="Cramers_V", ascending=False)

def calculate_cramers_v_for_ordinales_columns(X_train, y_train, ordinal_columns):
    results_cramer_cat = []
    for col in ordinal_columns:
        try:
            vcat = cramers_v(X_train[col], y_train["SINISTRE"])
            results_cramer_cat.append((col, vcat))
        except:
            continue
    return pd.DataFrame(results_cramer_cat, columns=["Variable_Ord", "Cramers_V"]).sort_values(by="Cramers_V", ascending=False)
