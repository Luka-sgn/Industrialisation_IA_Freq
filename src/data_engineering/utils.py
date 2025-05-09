def print_class_distribution(y_train):
    print("Répartition des classes :")
    print(y_train["SINISTRE"].value_counts())
