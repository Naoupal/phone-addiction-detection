from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow.xgboost, mlflow.tensorflow, mlflow.sklearn
import mlflow
import joblib, json, os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model

app = Flask(__name__)
CORS(app)

# ── Load konfigurasi ─────────────────────────────────────────
# Path mlruns sesuai lokasi project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLFLOW_TRACKING_URI = f"file:///{BASE_DIR}/mlruns".replace("\\", "/")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f"📂 Tracking URI: {MLFLOW_TRACKING_URI}")
print("⏳ Loading models dari MLflow Production...")

# ── Load models dengan fallback ke saved_models/ ─────────────
def load_models():
    global xgb_model, cnn_model, meta_model, extractor

    try:
        xgb_model  = mlflow.xgboost.load_model(
            "models:/SmartphoneAddiction_XGB_CNN_Augmented/Production")
        print("✅ XGBoost loaded dari MLflow Registry")
    except Exception as e:
        print(f"⚠️  MLflow Registry gagal ({e})")
        print("   → Fallback ke saved_models/xgb_cnn_augmented.pkl")
        xgb_model = joblib.load(os.path.join(BASE_DIR, "saved_models", "xgb_cnn_augmented.pkl"))

    try:
        cnn_model  = mlflow.tensorflow.load_model(
            "models:/SmartphoneAddiction_CNN_Extractor/Production")
        print("✅ CNN loaded dari MLflow Registry")
    except Exception as e:
        print(f"⚠️  MLflow Registry gagal ({e})")
        print("   → Fallback ke saved_models/cnn_extractor.keras")
        cnn_model = tf.keras.models.load_model(
            os.path.join(BASE_DIR, "saved_models", "cnn_extractor.keras"))

    try:
        meta_model = mlflow.sklearn.load_model(
            "models:/SmartphoneAddiction_MetaLearner/Production")
        print("✅ Meta-learner loaded dari MLflow Registry")
    except Exception as e:
        print(f"⚠️  MLflow Registry gagal ({e})")
        print("   → Fallback ke saved_models/meta_model.pkl")
        meta_model = joblib.load(os.path.join(BASE_DIR, "saved_models", "meta_model.pkl"))

    # Feature extractor dari CNN (layer sebelum softmax)
    extractor = Model(inputs=cnn_model.input,
                      outputs=cnn_model.layers[-3].output)
    print("✅ Semua model berhasil diload!")

load_models()

# ── Load pipeline artifacts ───────────────────────────────────
scaler    = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
col_order = joblib.load(os.path.join(BASE_DIR, "col_order.pkl"))

with open(os.path.join(BASE_DIR, "metadata.json")) as f:
    meta = json.load(f)

TARGET_NAMES = ["Mild", "Moderate", "Severe"]


def apply_feature_engineering(df):
    eps = 1e-5
    df = df.copy()
    df["social_ratio"]       = df["social_media_hours"] / (df["daily_screen_time_hours"] + eps)
    df["gaming_ratio"]       = df["gaming_hours"]       / (df["daily_screen_time_hours"] + eps)
    df["productive_ratio"]   = df["work_study_hours"]   / (df["daily_screen_time_hours"] + eps)
    df["passive_ratio"]      = (df["social_media_hours"] + df["gaming_hours"]) / (df["daily_screen_time_hours"] + eps)
    df["sleep_deficit"]      = np.maximum(0, 7 - df["sleep_hours"])
    df["sleep_screen_ratio"] = df["sleep_hours"] / (df["daily_screen_time_hours"] + eps)
    df["leisure_total"]      = df["social_media_hours"] + df["gaming_hours"]
    df["notif_density"]      = df["notifications_per_day"] / (df["daily_screen_time_hours"] + eps)
    df["app_open_rate"]      = df["app_opens_per_day"]     / (df["daily_screen_time_hours"] + eps)
    df["weekend_ratio"]      = df["weekend_screen_time"]   / (df["daily_screen_time_hours"] + eps)
    df["app_per_notif"]      = df["app_opens_per_day"]     / (df["notifications_per_day"] + eps)
    df["risk_score"] = (
        df["daily_screen_time_hours"] * 0.3 +
        df["social_media_hours"]      * 0.2 +
        df["sleep_deficit"]           * 0.25 +
        df["notif_density"] / 10      * 0.15 +
        df["gaming_hours"]            * 0.1
    )
    df["screen_time_bin"] = pd.cut(
        df["daily_screen_time_hours"],
        bins=[0, 2, 4, 6, 9, 24], labels=[0, 1, 2, 3, 4]).astype(float)
    df["sleep_quality"] = pd.cut(
        df["sleep_hours"],
        bins=[0, 5, 6, 7, 8, 24], labels=[0, 1, 2, 3, 4]).astype(float)
    return df


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service"  : "Smartphone Addiction Detector",
        "model"    : "Hybrid CNN + XGBoost",
        "status"   : "running",
        "endpoints": ["/predict", "/health"]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # ── Preprocessing ──────────────────────────────────────
        df_input = pd.DataFrame([data])
        df_input = apply_feature_engineering(df_input)

        missing = [f for f in meta["features"] if f not in df_input.columns]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        X_input  = df_input[meta["features"]].values
        X_scaled = scaler.transform(X_input)

        # ── Reorder semantik untuk CNN ─────────────────────────
        col_ord = [c for c in col_order if c in meta["features"]]
        feat_df = pd.DataFrame(X_scaled, columns=meta["features"])
        X_cnn   = feat_df[col_ord].values.reshape(-1, len(col_ord), 1)

        # ── Augmented features ─────────────────────────────────
        cnn_feat = extractor.predict(X_cnn, verbose=0)
        X_aug    = np.hstack([X_scaled, cnn_feat])

        # ── Stacking prediction ────────────────────────────────
        xgb_prob = xgb_model.predict_proba(X_aug)
        cnn_prob = cnn_model.predict(X_cnn, verbose=0)
        stack    = np.hstack([xgb_prob, cnn_prob])
        pred     = int(meta_model.predict(stack)[0])
        prob     = meta_model.predict_proba(stack)[0]

        label      = TARGET_NAMES[pred]
        confidence = float(prob[pred])

        interpretations = {
            "Mild"    : "Penggunaan smartphone masih dalam batas normal. Pertahankan kebiasaan baik.",
            "Moderate": "Terdapat tanda-tanda ketergantungan sedang. Mulai atur batas waktu layar.",
            "Severe"  : "Indikasi kecanduan kuat. Sangat disarankan mencari bantuan profesional.",
        }
        recommendations = {
            "Mild"    : ["Pertahankan kebiasaan screen time saat ini", "Tetap jaga sleep schedule"],
            "Moderate": ["Aktifkan Digital Wellbeing / Screen Time limiter", "Kurangi social media 1 jam/hari", "Jadwalkan phone-free hours"],
            "Severe"  : ["Konsultasi dengan psikolog atau konselor", "Gunakan aplikasi blocker", "Cari dukungan dari keluarga/teman"],
        }

        return jsonify({
            "prediction"      : label,
            "confidence"      : round(confidence, 4),
            "probabilities"   : {
                "Mild"    : round(float(prob[0]), 4),
                "Moderate": round(float(prob[1]), 4),
                "Severe"  : round(float(prob[2]), 4),
            },
            "interpretation"  : interpretations[label],
            "recommendations" : recommendations[label],
            "risk_score_input": round(float(
                data.get("daily_screen_time_hours", 0) * 0.3 +
                data.get("social_media_hours", 0) * 0.2), 4),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Flask API berjalan di http://localhost:5001")
    print("   Endpoint: POST /predict")
    app.run(host="0.0.0.0", port=5001, debug=False)
