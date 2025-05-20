from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import json
from typing import List

app = FastAPI(
    title="API de Prédiction FREQ",
    description="Upload d'un modèle XGBoost et d'un fichier .json contenant les variables pour prédire la fréquence",
    version="1.0"
)

# === Variables globales ===
model_path = "src/Model/params/best_model_freq_xgb.pkl"
model, columns_used = joblib.load(model_path)

@app.post("/predict")
async def predict_from_json(json_file: UploadFile = File(...)):
    """
    Prédit la fréquence à partir d'un fichier .json contenant les variables (format clé-valeur)
    """
    try:
        if model is None or columns_used is None:
            return {"error": "Le modèle n'est pas encore chargé. Appelez /load_model d'abord."}

        contents = await json_file.read()
        data_dict = json.loads(contents)
        df = pd.DataFrame([data_dict])

        df = df[columns_used]  # filtrage des colonnes attendues

        prediction = model.predict_proba(df)[:,1]
        return {"prediction_freq": float(prediction[0])}

    except Exception as e:
        return {"error": str(e)}
