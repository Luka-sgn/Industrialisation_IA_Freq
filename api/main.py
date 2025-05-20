from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from api.freq_predictor import FreqPredictor
from api.schemas import FreqPredictionInput

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API de Prédiction FREQ",
    description="API REST pour la prédiction de la fréquence de sinistres basée sur un modèle XGBoost optimisé avec Optuna",
    version="1.0"
)

# Chargement du modèle et de l’encodeur
model_path = "src/Model/params/best_model_freq_xgb.pkl"
model_freq = joblib.load(model_path)

# Initialisation du prédicteur
predictor = FreqPredictor()
predictor.model = model_freq

# === Routes ===
@app.get("/")
def root():
    return {"message": "API FREQ en ligne 🚀"}

@app.get("/health")
def health_check():
    return {"status": "OK", "message": "API opérationnelle (FREQ)"}

@app.post("/predict_freq")
def predict_freq(input_data: FreqPredictionInput):
    """
    Prédiction de la fréquence de sinistre à partir d’un input validé Pydantic
    """
    try:
        df = pd.DataFrame([input_data.dict()])
        df = FreqPredictor.reduce_memory_usage(df, verbose=False)

        y_pred = predictor.predict(
            df,
            df.select_dtypes(include="number").columns,
            df.select_dtypes(include="object").columns
        )

        return {"prediction_freq": float(y_pred[0])}

    except Exception as e:
        return {"error": str(e)}
