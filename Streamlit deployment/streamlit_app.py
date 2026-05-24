"""
========================================================================
Smartphone Addiction Detection - Streamlit Deployment
Model: Hybrid CNN (Feature Extractor) + XGBoost (Meta-Classifier)
========================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model, Model

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="📱 Deteksi Kecanduan Smartphone",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 2rem;
    text-align: center;
}
.main-header h1 {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.main-header p {
    font-size: 1rem;
    opacity: 0.9;
}
.result-card {
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    margin: 1rem 0;
    color: white;
    font-weight: 600;
}
.result-mild {
    background: linear-gradient(135deg, #56ab2f, #a8e063);
}
.result-moderate {
    background: linear-gradient(135deg, #f7971e, #ffd200);
    color: #333;
}
.result-severe {
    background: linear-gradient(135deg, #cb2d3e, #ef473a);
}
.metric-card {
    background: #f8f9fa;
    padding: 1.2rem;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #e9ecef;
}
.metric-card h3 {
    font-size: 0.85rem;
    color: #6c757d;
    margin-bottom: 0.3rem;
}
.metric-card p {
    font-size: 1.4rem;
    font-weight: 700;
    color: #333;
    margin: 0;
}
.info-box {
    background: #e8f4fd;
    border-left: 4px solid #2196F3;
    padding: 1rem 1.5rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}
.warning-box {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 1rem 1.5rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}
.danger-box {
    background: #f8d7da;
    border-left: 4px solid #dc3545;
    padding: 1rem 1.5rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Models ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load all saved models and artifacts"""
    model_dir = "saved_models"

    xgb_model  = joblib.load(os.path.join(model_dir, "xgb_cnn_augmented.pkl"))
    meta_model = joblib.load(os.path.join(model_dir, "meta_model.pkl"))
    scaler     = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    col_order  = joblib.load(os.path.join(model_dir, "col_order.pkl"))

    cnn_model  = load_model(os.path.join(model_dir, "cnn_extractor.keras"))

    with open(os.path.join(model_dir, "metadata.json")) as f:
        metadata = json.load(f)

    # Build feature extractor (layer before softmax)
    extractor = Model(
        inputs=cnn_model.inputs,
        outputs=cnn_model.layers[-3].output
    )

    return xgb_model, cnn_model, meta_model, scaler, col_order, metadata, extractor


def apply_feature_engineering(df):
    """Apply the same feature engineering as training pipeline"""
    eps = 1e-5
    df = df.copy()

    # Group A: Usage Ratios
    df["social_ratio"]     = df["social_media_hours"] / (df["daily_screen_time_hours"] + eps)
    df["gaming_ratio"]     = df["gaming_hours"]       / (df["daily_screen_time_hours"] + eps)
    df["productive_ratio"] = df["work_study_hours"]   / (df["daily_screen_time_hours"] + eps)
    df["passive_ratio"]    = (df["social_media_hours"] + df["gaming_hours"]) / (df["daily_screen_time_hours"] + eps)

    # Group B: Wellbeing Indicators
    df["sleep_deficit"]      = np.maximum(0, 7 - df["sleep_hours"])
    df["sleep_screen_ratio"] = df["sleep_hours"] / (df["daily_screen_time_hours"] + eps)
    df["leisure_total"]      = df["social_media_hours"] + df["gaming_hours"]

    # Group C: Behavioral Intensity
    df["notif_density"]  = df["notifications_per_day"] / (df["daily_screen_time_hours"] + eps)
    df["app_open_rate"]  = df["app_opens_per_day"]     / (df["daily_screen_time_hours"] + eps)
    df["weekend_ratio"]  = df["weekend_screen_time"]   / (df["daily_screen_time_hours"] + eps)
    df["app_per_notif"]  = df["app_opens_per_day"]     / (df["notifications_per_day"] + eps)

    # Group D: Composite Risk Score
    df["risk_score"] = (
        df["daily_screen_time_hours"] * 0.3 +
        df["social_media_hours"]      * 0.2 +
        df["sleep_deficit"]           * 0.25 +
        df["notif_density"] / 10      * 0.15 +
        df["gaming_hours"]            * 0.1
    )

    # Group E: Binned Features
    df["screen_time_bin"] = pd.cut(
        df["daily_screen_time_hours"],
        bins=[0, 2, 4, 6, 9, 24], labels=[0, 1, 2, 3, 4]
    ).astype(float)
    df["sleep_quality"] = pd.cut(
        df["sleep_hours"],
        bins=[0, 5, 6, 7, 8, 24], labels=[0, 1, 2, 3, 4]
    ).astype(float)

    return df


def predict_addiction(input_data, xgb_model, cnn_model, meta_model,
                    scaler, col_order, metadata, extractor):
    """Run the full hybrid prediction pipeline"""
    # Create DataFrame
    df_input = pd.DataFrame([input_data])

    # Feature engineering
    df_input = apply_feature_engineering(df_input)

    # Encode gender
    gender_map = {"Male": 1, "Female": 0, "Other": 2}
    if "gender" in df_input.columns:
        df_input["gender"] = df_input["gender"].map(gender_map).fillna(2)

    # Encode stress_level
    stress_map = {"Low": 0, "Medium": 1, "High": 2}
    if "stress_level" in df_input.columns:
        df_input["stress_level"] = df_input["stress_level"].map(stress_map).fillna(1)

    # Encode academic_work_impact
    impact_map = {"No": 0, "Yes": 1}
    if "academic_work_impact" in df_input.columns:
        df_input["academic_work_impact"] = df_input["academic_work_impact"].map(impact_map).fillna(0)

    # Select features used in training
    features = metadata["features"]
    available = [f for f in features if f in df_input.columns]
    missing   = [f for f in features if f not in df_input.columns]

    if missing:
        for m in missing:
            df_input[m] = 0

    X_input  = df_input[features].values.astype(float)
    X_input  = np.nan_to_num(X_input, nan=0.0)
    X_scaled = scaler.transform(X_input)

    # Reorder for CNN
    feat_df  = pd.DataFrame(X_scaled, columns=features)
    col_ord  = [c for c in col_order if c in features]
    remaining = [c for c in features if c not in col_ord]
    col_ord  += remaining
    X_cnn    = feat_df[col_ord].values.reshape(-1, len(col_ord), 1)

    # Extract CNN features
    cnn_feat = extractor.predict(X_cnn, verbose=0)
    X_aug    = np.hstack([X_scaled, cnn_feat])

    # Stacking prediction
    xgb_prob = xgb_model.predict_proba(X_aug)
    cnn_prob = cnn_model.predict(X_cnn, verbose=0)
    stack    = np.hstack([xgb_prob, cnn_prob])

    pred = int(meta_model.predict(stack)[0])
    prob = meta_model.predict_proba(stack)[0]

    target_names = ["Mild", "Moderate", "Severe"]
    label      = target_names[pred]
    confidence = float(prob[pred])

    return label, confidence, {
        "Mild": float(prob[0]),
        "Moderate": float(prob[1]),
        "Severe": float(prob[2])
    }, {
        "xgb_prob": xgb_prob[0].tolist(),
        "cnn_prob": cnn_prob[0].tolist()
    }


# ─── Main App ────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📱 Deteksi Kecanduan Penggunaan Smartphone</h1>
        <p>Menggunakan Arsitektur Hybrid CNN (Feature Extractor) + XGBoost (Meta-Classifier)</p>
    </div>
    """, unsafe_allow_html=True)

    # Try to load models
    try:
        xgb_model, cnn_model, meta_model, scaler, col_order, metadata, extractor = load_models()
        models_loaded = True
    except Exception as e:
        models_loaded = False
        st.error(f"⚠️ Model belum tersedia. Jalankan notebook training terlebih dahulu. Error: {e}")

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Pengaturan")
        mode = st.radio("Mode Input:", ["📝 Manual Input", "📁 Upload CSV"], index=0)

        st.markdown("---")
        st.markdown("### 📊 Tentang Model")
        st.markdown("""
        **Arsitektur:** Hybrid CNN + XGBoost

        **Pipeline:**
        1. CNN sebagai Feature Extractor
        2. XGBoost pada augmented features
        3. Meta-Learner (Stacking)

        **Target:**
        - 🟢 Mild (Ringan)
        - 🟡 Moderate (Sedang)
        - 🔴 Severe (Berat)
        """)

        st.markdown("---")
        st.markdown("### 👨‍💻 Informasi Proyek")
        st.markdown("""
        **Mata Kuliah:** Data Mining
        **Algoritma:** CNN × XGBoost
        **Dataset:** 7.500 records
        """)

    # ─── Manual Input Mode ────────────────────────────────────
    if mode == "📝 Manual Input":
        st.markdown("### 📋 Masukkan Data Penggunaan Smartphone")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Data Demografi")
            age = st.number_input("Usia", min_value=5, max_value=100, value=22, step=1)
            gender = st.selectbox("Jenis Kelamin", ["Male", "Female", "Other"])

        with col2:
            st.markdown("##### 📱 Penggunaan Harian")
            daily_screen_time = st.slider("Screen Time Harian (jam)", 0.0, 24.0, 6.0, 0.5)
            social_media = st.slider("Media Sosial (jam)", 0.0, 24.0, 3.0, 0.5)
            gaming = st.slider("Gaming (jam)", 0.0, 24.0, 1.5, 0.5)
            work_study = st.slider("Kerja/Belajar (jam)", 0.0, 24.0, 3.0, 0.5)

        with col3:
            st.markdown("##### 😴 Pola Hidup")
            sleep_hours = st.slider("Jam Tidur", 0.0, 24.0, 7.0, 0.5)
            notifications = st.number_input("Notifikasi/Hari", 0, 500, 120, 10)
            app_opens = st.number_input("Buka Aplikasi/Hari", 0, 500, 80, 5)
            weekend_screen = st.slider("Screen Time Weekend (jam)", 0.0, 24.0, 8.0, 0.5)

        col_a, col_b = st.columns(2)
        with col_a:
            stress_level = st.selectbox("Tingkat Stres", ["Low", "Medium", "High"])
        with col_b:
            academic_impact = st.selectbox("Dampak pada Akademik/Kerja?", ["Yes", "No"])

        st.markdown("---")

        # Predict button
        if st.button("🔍 Analisis Tingkat Kecanduan", type="primary", use_container_width=True):
            if not models_loaded:
                st.error("Model belum tersedia!")
                return

            input_data = {
                "age": age,
                "gender": gender,
                "daily_screen_time_hours": daily_screen_time,
                "social_media_hours": social_media,
                "gaming_hours": gaming,
                "work_study_hours": work_study,
                "sleep_hours": sleep_hours,
                "notifications_per_day": notifications,
                "app_opens_per_day": app_opens,
                "weekend_screen_time": weekend_screen,
                "stress_level": stress_level,
                "academic_work_impact": academic_impact,
            }

            with st.spinner("🔄 Memproses prediksi..."):
                label, confidence, probs, sub_probs = predict_addiction(
                    input_data, xgb_model, cnn_model, meta_model,
                    scaler, col_order, metadata, extractor
                )

            # ─── Display Results ──────────────────────────────
            st.markdown("---")
            st.markdown("## 📊 Hasil Analisis")

            # Result card
            css_class = {
                "Mild": "result-mild",
                "Moderate": "result-moderate",
                "Severe": "result-severe"
            }[label]

            emoji = {"Mild": "🟢", "Moderate": "🟡", "Severe": "🔴"}[label]
            level_id = {"Mild": "Ringan", "Moderate": "Sedang", "Severe": "Berat"}[label]

            st.markdown(f"""
            <div class="result-card {css_class}">
                <h2>{emoji} Tingkat Kecanduan: {label} ({level_id})</h2>
                <p style="font-size:1.2rem;">Confidence: {confidence*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            # Probability gauge
            col1, col2, col3 = st.columns(3)
            for col, (name, prob_val) in zip(
                [col1, col2, col3],
                [("🟢 Mild", probs["Mild"]),
                 ("🟡 Moderate", probs["Moderate"]),
                 ("🔴 Severe", probs["Severe"])]
            ):
                with col:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob_val * 100,
                        title={"text": name},
                        number={"suffix": "%"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#667eea"},
                            "steps": [
                                {"range": [0, 33], "color": "#e8f5e9"},
                                {"range": [33, 66], "color": "#fff3e0"},
                                {"range": [66, 100], "color": "#ffebee"},
                            ]
                        }
                    ))
                    fig.update_layout(height=250, margin=dict(t=50, b=10, l=30, r=30))
                    st.plotly_chart(fig, use_container_width=True)

            # Probability bar chart
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=["Mild (Ringan)", "Moderate (Sedang)", "Severe (Berat)"],
                    y=[probs["Mild"]*100, probs["Moderate"]*100, probs["Severe"]*100],
                    marker_color=["#4CAF50", "#FF9800", "#F44336"],
                    text=[f"{v*100:.1f}%" for v in [probs["Mild"], probs["Moderate"], probs["Severe"]]],
                    textposition="auto"
                )
            ])
            fig_bar.update_layout(
                title="Distribusi Probabilitas Prediksi",
                yaxis_title="Probabilitas (%)",
                height=350,
                template="plotly_white"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Sub-model comparison
            st.markdown("#### 🔬 Perbandingan Sub-Model")
            col1, col2 = st.columns(2)
            with col1:
                fig_xgb = go.Figure(data=[go.Bar(
                    x=["Mild", "Moderate", "Severe"],
                    y=[v*100 for v in sub_probs["xgb_prob"]],
                    marker_color=["#4CAF50", "#FF9800", "#F44336"]
                )])
                fig_xgb.update_layout(title="XGBoost + CNN Features", height=300,
                                    yaxis_title="%", template="plotly_white")
                st.plotly_chart(fig_xgb, use_container_width=True)
            with col2:
                fig_cnn = go.Figure(data=[go.Bar(
                    x=["Mild", "Moderate", "Severe"],
                    y=[v*100 for v in sub_probs["cnn_prob"]],
                    marker_color=["#4CAF50", "#FF9800", "#F44336"]
                )])
                fig_cnn.update_layout(title="CNN Feature Extractor", height=300,
                                    yaxis_title="%", template="plotly_white")
                st.plotly_chart(fig_cnn, use_container_width=True)

            # Interpretation & Recommendations
            interpretations = {
                "Mild":     "Penggunaan smartphone Anda masih dalam batas normal. Pertahankan kebiasaan baik Anda.",
                "Moderate": "Terdapat tanda-tanda ketergantungan sedang. Mulai atur batas waktu penggunaan layar.",
                "Severe":   "Indikasi kecanduan kuat terdeteksi. Sangat disarankan untuk mencari bantuan profesional.",
            }
            recommendations = {
                "Mild": [
                    "✅ Pertahankan kebiasaan screen time saat ini",
                    "✅ Tetap jaga jadwal tidur yang teratur",
                    "✅ Lanjutkan aktivitas produktif di luar smartphone",
                ],
                "Moderate": [
                    "⚠️ Aktifkan fitur Digital Wellbeing / Screen Time limiter",
                    "⚠️ Kurangi penggunaan media sosial 1 jam/hari",
                    "⚠️ Jadwalkan phone-free hours (misal saat makan)",
                    "⚠️ Gunakan mode Do Not Disturb saat tidur",
                ],
                "Severe": [
                    "🚨 Konsultasi dengan psikolog atau konselor profesional",
                    "🚨 Gunakan aplikasi blocker untuk membatasi akses",
                    "🚨 Cari dukungan dari keluarga dan teman terdekat",
                    "🚨 Tetapkan jadwal digital detox secara rutin",
                    "🚨 Ganti aktivitas smartphone dengan hobi offline",
                ],
            }

            box_class = {"Mild": "info-box", "Moderate": "warning-box", "Severe": "danger-box"}[label]
            st.markdown(f"""
            <div class="{box_class}">
                <strong>💡 Interpretasi:</strong><br>{interpretations[label]}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📝 Rekomendasi")
            for rec in recommendations[label]:
                st.markdown(f"- {rec}")

            # Input summary radar chart
            st.markdown("#### 📈 Profil Penggunaan Anda")
            categories = ["Screen Time", "Sosial Media", "Gaming",
                        "Kerja/Belajar", "Tidur", "Notifikasi"]
            # Normalize values for radar (0-1 scale)
            norm_vals = [
                daily_screen_time / 24,
                social_media / 24,
                gaming / 24,
                work_study / 24,
                sleep_hours / 24,
                notifications / 500,
            ]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=norm_vals + [norm_vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor="rgba(102,126,234,0.2)",
                line_color="#667eea",
                name="Profil Anda"
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False, height=400, title="Radar Profil Penggunaan"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # ─── CSV Upload Mode ──────────────────────────────────────
    elif mode == "📁 Upload CSV":
        st.markdown("### 📁 Upload File CSV untuk Prediksi Batch")

        st.markdown("""
        Upload file CSV dengan kolom berikut:
        `age`, `gender`, `daily_screen_time_hours`, `social_media_hours`,
        `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`,
        `app_opens_per_day`, `weekend_screen_time`, `stress_level`, `academic_work_impact`
        """)

        uploaded = st.file_uploader("Pilih file CSV", type=["csv"])

        if uploaded and models_loaded:
            df = pd.read_csv(uploaded)
            st.markdown(f"**Data Preview** ({len(df)} baris)")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("🔍 Prediksi Semua Data", type="primary", use_container_width=True):
                results = []
                progress = st.progress(0)

                for i, row in df.iterrows():
                    input_data = row.to_dict()
                    try:
                        label, confidence, probs, _ = predict_addiction(
                            input_data, xgb_model, cnn_model, meta_model,
                            scaler, col_order, metadata, extractor
                        )
                        results.append({
                            "index": i,
                            "prediction": label,
                            "confidence": confidence,
                            "prob_mild": probs["Mild"],
                            "prob_moderate": probs["Moderate"],
                            "prob_severe": probs["Severe"],
                        })
                    except Exception as e:
                        results.append({
                            "index": i,
                            "prediction": "Error",
                            "confidence": 0,
                            "prob_mild": 0,
                            "prob_moderate": 0,
                            "prob_severe": 0,
                        })
                    progress.progress((i + 1) / len(df))

                results_df = pd.DataFrame(results)

                st.markdown("---")
                st.markdown("### 📊 Hasil Prediksi Batch")
                st.dataframe(results_df, use_container_width=True)

                # Summary
                counts = results_df["prediction"].value_counts()
                fig_pie = px.pie(
                    values=counts.values,
                    names=counts.index,
                    color=counts.index,
                    color_discrete_map={
                        "Mild": "#4CAF50",
                        "Moderate": "#FF9800",
                        "Severe": "#F44336",
                        "Error": "#999"
                    },
                    title="Distribusi Hasil Prediksi"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                # Download results
                csv_out = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Hasil Prediksi (CSV)",
                    csv_out,
                    "prediction_results.csv",
                    "text/csv",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()
