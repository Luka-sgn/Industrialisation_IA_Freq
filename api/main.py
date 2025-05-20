from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import json
from api.freq_predictor import FreqPredictor
from api.preprocessing import Preprocessor

# === Initialisation ===
app = FastAPI(
    title="API de Prédiction FREQ",
    description="Upload d'un fichier .json contenant les variables pour prédire la fréquence",
    version="1.0"
)

# === Chargement du modèle ===
model_path = "src/Model/params/best_model_freq_xgb.pkl"
model, columns_used = joblib.load(model_path)

# === Initialisation des classes utiles ===
predictor = FreqPredictor()
predictor.model = model
preproc = Preprocessor()

@app.post("/predict")
async def predict_from_json(json_file: UploadFile = File(...)):
    """
    Prédit la fréquence à partir d'un fichier .json contenant les variables (clé-valeur)
    """
    try:
        # Lecture et parsing
        contents = await json_file.read()
        data_dict = json.loads(contents)
        df = pd.DataFrame([data_dict])

        # Nettoyage via Preprocessor
        df_clean = preproc.clean_dataframe(df)

        # Réduction mémoire
        df_clean = FreqPredictor.reduce_memory_usage(df_clean, verbose=False)

        # Filtrage des colonnes utilisées
        df_filtered = df_clean[columns_used]

        # Prédiction
        prediction = predictor.model.predict_proba(df_filtered)[:, 1]

        return {"prediction_freq": float(prediction[0])}

    except Exception as e:
        return {"error": str(e)}