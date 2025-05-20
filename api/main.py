from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import json
from api.freq_predictor import FreqPredictor
from api.preprocessing import Preprocessor
from api.cm_predictor import CMPredictor
from api.schemas import CMPredictionInput  # le schéma généré avec toutes les variables en int

# === Initialisation ===
app = FastAPI(
    title="API de Prédiction Sinistres",
    description="Upload d'un fichier .json contenant les variables pour prédire la fréquence, le coût moyen (CM), et le coût attendu",
    version="2.0"
)

# === Chargement des modèles ===
model_freq_path = "src/Model/params/best_model_freq_xgb.pkl"
model_cm_path = "src/Model/params/best_model_encoder_cm_xgb.pkl"

model_freq, columns_used_freq = joblib.load(model_freq_path)
model_cm, columns_used_cm = joblib.load(model_cm_path)

# === Initialisation des classes utiles ===
freq_predictor = FreqPredictor()
freq_predictor.model = model_freq

cm_predictor = CMPredictor()
cm_predictor.model = model_cm

preproc = Preprocessor()

# === Route de prédiction de la fréquence ===
@app.post("/predict_freq")
async def predict_freq(json_file: UploadFile = File(...)):
    try:
        contents = await json_file.read()
        data_dict = json.loads(contents)
        df = pd.DataFrame([data_dict])

        df_clean = preproc.clean_dataframe(df)
        df_filtered = df_clean[columns_used_freq]

        prediction = freq_predictor.model.predict_proba(df_filtered)[:, 1]
        return {"prediction_freq": float(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_cm")
async def predict_cm(json_file: UploadFile = File(...)):
    try:
        contents = await json_file.read()
        data_dict = json.loads(contents)
        df = pd.DataFrame([data_dict])

        df_clean = preproc.clean_dataframe(df)
        df_filtered = df_clean[columns_used_cm]

        prediction = cm_predictor.model.predict(df_filtered)
        return {"prediction_cm": float(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

# === Route de prédiction du coût total estimé (fréquence * coût moyen) ===
@app.post("/predict_total_cost")
async def predict_total_cost(json_file: UploadFile = File(...)):
    try:
        contents = await json_file.read()
        data_dict = json.loads(contents)
        df = pd.DataFrame([data_dict])

        df_clean = preproc.clean_dataframe(df)

        df_freq = df_clean[columns_used_freq]
        df_cm = df_clean[columns_used_cm]

        pred_freq = freq_predictor.model.predict_proba(df_freq)[:, 1][0]
        pred_cm = cm_predictor.model.predict(df_cm)[0]

        estimated_cost = pred_freq * pred_cm
        return {
            "prediction_freq": float(pred_freq),
            "prediction_cm": float(pred_cm),
            "prediction_total_cost": float(estimated_cost)
        }
    except Exception as e:
        return {"error": str(e)}
