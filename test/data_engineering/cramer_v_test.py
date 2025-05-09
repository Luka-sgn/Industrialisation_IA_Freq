import pandas as pd
from src.data_engineering.cramer_v import cramers_v

def test_cramers_v():
    # Données d'exemple
    data = {'col1': ['A', 'B', 'A', 'B', 'A'], 'col2': ['X', 'X', 'Y', 'Y', 'X']}
    df = pd.DataFrame(data)

    # Calcul de Cramér's V
    result = cramers_v(df['col1'], df['col2'])

    # Vérifier que le résultat est un nombre
    assert isinstance(result, float)
    assert result >= 0 and result <= 1  # Cramér's V doit être entre 0 et 1
