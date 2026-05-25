"""
========================================================================
Smartphone Addiction Detection - Streamlit Deployment
Model: Hybrid CNN (Feature Extractor) + XGBoost (Meta-Classifier)
Target: addicted_label (Binary: 0 = Tidak Kecanduan, 1 = Kecanduan)
========================================================================
Jalankan: streamlit run streamlit_app.py
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
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem; border-radius: 16px; color: white;
    margin-bottom: 2rem; text-align: center;
}
.main-header h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
.main-header p { font-size: 1rem; opacity: 0.9; }
.result-card {
    padding: 2rem; border-radius: 16px; text-align: center;
    margin: 1rem 0; color: white; font-weight: 600;
}
.result-safe { background: linear-gradient(135deg, #56ab2f, #a8e063); }
.result-addicted { background: linear-gradient(135deg, #cb2d3e, #ef473a); }
.info-box {
    background: #e8f4fd; border-left: 4px solid #2196F3;
    padding: 1rem 1.5rem; border-radius: 0 8px 8px 0; margin: 1rem 0;
}
.danger-box {
    background: #f8d7da; border-left: 4px solid #dc3545;
    padding: 1rem 1.5rem; border-radius: 0 8px 8px 0; margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Models ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Folder '{model_dir}' tidak ditemukan. "
            "Jalankan train_model.py terlebih dahulu."
        )

    xgb_model = joblib.load(os.path.join(model_dir, "xgb_cnn_augmented.pkl"))
    meta_model = joblib.load(os.path.join(model_dir, "meta_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    col_order = joblib.load(os.path.join(model_dir, "col_order.pkl"))
    cnn_model = load_model(os.path.join(model_dir, "cnn_extractor.keras"))

    with open(os.path.join(model_dir, "metadata.json")) as f:
        metadata = json.load(f)

    extractor = Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[-3].output)
    return xgb_model, cnn_model, meta_model, scaler, col_order, metadata, extractor


def apply_feature_engineering(df):
    eps = 1e-5
    df = df.copy()

    df["social_ratio"] = df["social_media_hours"] / (df["daily_screen_time_hours"] + eps)
    df["gaming_ratio"] = df["gaming_hours"] / (df["daily_screen_time_hours"] + eps)
    df["productive_ratio"] = df["work_study_hours"] / (df["daily_screen_time_hours"] + eps)
    df["passive_ratio"] = (df["social_media_hours"] + df["gaming_hours"]) / (df["daily_screen_time_hours"] + eps)

    df["sleep_deficit"] = np.maximum(0, 7 - df["sleep_hours"])
    df["sleep_screen_ratio"] = df["sleep_hours"] / (df["daily_screen_time_hours"] + eps)
    df["leisure_total"] = df["social_media_hours"] + df["gaming_hours"]

    df["notif_density"] = df["notifications_per_day"] / (df["daily_screen_time_hours"] + eps)
    df["app_open_rate"] = df["app_opens_per_day"] / (df["daily_screen_time_hours"] + eps)
    df["weekend_ratio"] = df["weekend_screen_time"] / (df["daily_screen_time_hours"] + eps)
    df["app_per_notif"] = df["app_opens_per_day"] / (df["notifications_per_day"] + eps)

    df["risk_score"] = (
        df["daily_screen_time_hours"] * 0.3 + df["social_media_hours"] * 0.2 +
        df["sleep_deficit"] * 0.25 + df["notif_density"] / 10 * 0.15 +
        df["gaming_hours"] * 0.1
    )

    df["screen_time_bin"] = pd.cut(
        df["daily_screen_time_hours"],
        bins=[-0.001, 2, 4, 6, 9, 24], labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(2.0)
    df["sleep_quality"] = pd.cut(
        df["sleep_hours"],
        bins=[-0.001, 5, 6, 7, 8, 24], labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(2.0)

    return df


def encode_categoricals(df):
    df = df.copy()
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female": 0, "Male": 1, "Other": 2}).fillna(2).astype(float)
    if "stress_level" in df.columns:
        df["stress_level"] = df["stress_level"].map({"Low": 0, "Medium": 1, "High": 2}).fillna(1).astype(float)
    if "academic_work_impact" in df.columns:
        df["academic_work_impact"] = df["academic_work_impact"].map({"No": 0, "Yes": 1}).fillna(0).astype(float)
    return df


def predict_single(data, xgb_model, cnn_model, meta_model, scaler, col_order, metadata, extractor):
    df = pd.DataFrame([data])
    df = apply_feature_engineering(df)
    df = encode_categoricals(df)

    features = metadata["features"]
    for f in features:
        if f not in df.columns:
            df[f] = 0.0

    X = df[features].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)
    X_sc = scaler.transform(X)

    feat_df = pd.DataFrame(X_sc, columns=features)
    col_ord = [c for c in col_order if c in features]
    col_ord += [c for c in features if c not in col_ord]
    X_cnn = feat_df[col_ord].values.reshape(-1, len(col_ord), 1)

    cnn_feat = extractor.predict(X_cnn, verbose=0)
    X_aug = np.hstack([X_sc, cnn_feat])

    xgb_prob = xgb_model.predict_proba(X_aug)
    cnn_prob = cnn_model.predict(X_cnn, verbose=0)
    stack = np.hstack([xgb_prob, cnn_prob])

    pred = int(meta_model.predict(stack)[0])
    prob = meta_model.predict_proba(stack)[0]

    names = ["Tidak Kecanduan", "Kecanduan"]
    return names[pred], float(prob[pred]), {
        "Tidak Kecanduan": float(prob[0]),
        "Kecanduan": float(prob[1])
    }, {"xgb_prob": xgb_prob[0].tolist(), "cnn_prob": cnn_prob[0].tolist()}


def predict_batch(df, xgb_model, cnn_model, meta_model, scaler, col_order, metadata, extractor):
    df_in = apply_feature_engineering(df.copy())
    df_in = encode_categoricals(df_in)
    features = metadata["features"]
    for f in features:
        if f not in df_in.columns:
            df_in[f] = 0.0

    X = df_in[features].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)
    X_sc = scaler.transform(X)

    feat_df = pd.DataFrame(X_sc, columns=features)
    col_ord = [c for c in col_order if c in features]
    col_ord += [c for c in features if c not in col_ord]
    X_cnn = feat_df[col_ord].values.reshape(-1, len(col_ord), 1)

    cnn_feat = extractor.predict(X_cnn, verbose=0)
    X_aug = np.hstack([X_sc, cnn_feat])

    xgb_prob = xgb_model.predict_proba(X_aug)
    cnn_prob = cnn_model.predict(X_cnn, verbose=0)
    stack = np.hstack([xgb_prob, cnn_prob])

    preds = meta_model.predict(stack)
    probs = meta_model.predict_proba(stack)

    names = ["Tidak Kecanduan", "Kecanduan"]
    rows = []
    for i in range(len(preds)):
        label = names[int(preds[i])]
        rows.append({
            "index": i, "prediction": label,
            "confidence": float(probs[i][int(preds[i])]),
            "prob_tidak_kecanduan": float(probs[i][0]),
            "prob_kecanduan": float(probs[i][1]),
        })
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="main-header">
        <h1>📱 Deteksi Kecanduan Penggunaan Smartphone</h1>
        <p>Arsitektur Hybrid CNN (Feature Extractor) + XGBoost (Meta-Classifier)</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        xgb_model, cnn_model, meta_model, scaler, col_order, metadata, extractor = load_models()
        models_loaded = True
    except Exception as e:
        models_loaded = False
        st.error(f"""
        ⚠️ **Model belum tersedia.** Jalankan training:
        ```
        python train_model.py
        ```
        Error: {e}
        """)

    with st.sidebar:
        st.markdown("### ⚙️ Pengaturan")
        mode = st.radio("Mode Input:", ["📝 Manual Input", "📁 Upload CSV"])
        st.markdown("---")
        st.markdown("### 📊 Tentang Model")
        st.markdown("""
        **Arsitektur:** Hybrid CNN + XGBoost

        **Pipeline:**
        1. CNN sebagai Feature Extractor
        2. XGBoost pada augmented features
        3. Meta-Learner (Stacking)

        **Target (Binary):**
        - 🟢 Tidak Kecanduan (0)
        - 🔴 Kecanduan (1)
        """)
        st.markdown("---")
        st.markdown("### 👨‍💻 Informasi Proyek")
        st.markdown("**Mata Kuliah:** Data Mining\n\n**Algoritma:** CNN × XGBoost\n\n**Dataset:** 7.500 records")

    if mode == "📝 Manual Input":
        st.markdown("### 📋 Masukkan Data Penggunaan Smartphone")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### 👤 Demografi")
            age = st.number_input("Usia", 5, 100, 22)
            gender = st.selectbox("Jenis Kelamin", ["Male", "Female", "Other"])
        with col2:
            st.markdown("##### 📱 Penggunaan Harian")
            screen_time = st.slider("Screen Time (jam)", 0.0, 24.0, 6.0, 0.5)
            social = st.slider("Media Sosial (jam)", 0.0, 24.0, 3.0, 0.5)
            gaming = st.slider("Gaming (jam)", 0.0, 24.0, 1.5, 0.5)
            work = st.slider("Kerja/Belajar (jam)", 0.0, 24.0, 3.0, 0.5)
        with col3:
            st.markdown("##### 😴 Pola Hidup")
            sleep = st.slider("Jam Tidur", 0.0, 24.0, 7.0, 0.5)
            notif = st.number_input("Notifikasi/Hari", 0, 500, 120, 10)
            app_opens = st.number_input("Buka App/Hari", 0, 500, 80, 5)
            weekend = st.slider("Weekend Screen Time (jam)", 0.0, 24.0, 8.0, 0.5)

        ca, cb = st.columns(2)
        with ca:
            stress = st.selectbox("Tingkat Stres", ["Low", "Medium", "High"])
        with cb:
            impact = st.selectbox("Dampak Akademik/Kerja?", ["Yes", "No"])

        total_active = social + gaming + work
        if total_active > 24:
            st.warning(f"⚠️ Total jam aktif ({total_active:.1f}h) > 24 jam.")

        st.markdown("---")

        if st.button("🔍 Analisis Kecanduan", type="primary", use_container_width=True):
            if not models_loaded:
                st.error("Model belum tersedia!")
                return

            data = {
                "age": age, "gender": gender,
                "daily_screen_time_hours": screen_time,
                "social_media_hours": social, "gaming_hours": gaming,
                "work_study_hours": work, "sleep_hours": sleep,
                "notifications_per_day": notif, "app_opens_per_day": app_opens,
                "weekend_screen_time": weekend,
                "stress_level": stress, "academic_work_impact": impact,
            }

            with st.spinner("🔄 Memproses..."):
                try:
                    label, conf, probs, sub = predict_single(
                        data, xgb_model, cnn_model, meta_model,
                        scaler, col_order, metadata, extractor)
                except Exception as e:
                    st.error(f"Error: {e}")
                    return

            st.markdown("---")
            st.markdown("## 📊 Hasil Analisis")

            css = "result-safe" if label == "Tidak Kecanduan" else "result-addicted"
            emoji = "🟢" if label == "Tidak Kecanduan" else "🔴"

            st.markdown(f"""
            <div class="result-card {css}">
                <h2>{emoji} {label}</h2>
                <p style="font-size:1.2rem;">Confidence: {conf*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            # Gauge
            c1, c2 = st.columns(2)
            for col, (nm, val) in zip([c1, c2],
                [("🟢 Tidak Kecanduan", probs["Tidak Kecanduan"]),
                 ("🔴 Kecanduan", probs["Kecanduan"])]):
                with col:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=val * 100,
                        title={"text": nm}, number={"suffix": "%"},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#667eea"},
                               "steps": [{"range": [0, 50], "color": "#e8f5e9"},
                                          {"range": [50, 100], "color": "#ffebee"}]}
                    ))
                    fig.update_layout(height=250, margin=dict(t=50, b=10, l=30, r=30))
                    st.plotly_chart(fig, use_container_width=True)

            # Bar
            fig_bar = go.Figure(data=[go.Bar(
                x=["Tidak Kecanduan", "Kecanduan"],
                y=[probs["Tidak Kecanduan"]*100, probs["Kecanduan"]*100],
                marker_color=["#4CAF50", "#F44336"],
                text=[f"{v*100:.1f}%" for v in [probs["Tidak Kecanduan"], probs["Kecanduan"]]],
                textposition="auto"
            )])
            fig_bar.update_layout(title="Probabilitas Prediksi", yaxis_title="%",
                                  height=350, template="plotly_white")
            st.plotly_chart(fig_bar, use_container_width=True)

            # Sub-model
            st.markdown("#### 🔬 Perbandingan Sub-Model")
            c1, c2 = st.columns(2)
            with c1:
                fig_x = go.Figure(data=[go.Bar(x=["Tidak Kecanduan", "Kecanduan"],
                    y=[v*100 for v in sub["xgb_prob"]], marker_color=["#4CAF50", "#F44336"])])
                fig_x.update_layout(title="XGBoost + CNN Features", height=300, yaxis_title="%", template="plotly_white")
                st.plotly_chart(fig_x, use_container_width=True)
            with c2:
                fig_c = go.Figure(data=[go.Bar(x=["Tidak Kecanduan", "Kecanduan"],
                    y=[v*100 for v in sub["cnn_prob"]], marker_color=["#4CAF50", "#F44336"])])
                fig_c.update_layout(title="CNN Extractor", height=300, yaxis_title="%", template="plotly_white")
                st.plotly_chart(fig_c, use_container_width=True)

            # Interpretasi
            interp = {
                "Tidak Kecanduan": "Penggunaan smartphone Anda masih dalam batas wajar. Pertahankan kebiasaan baik ini.",
                "Kecanduan": "Terdeteksi indikasi kecanduan smartphone. Disarankan untuk mengurangi waktu penggunaan.",
            }
            recs = {
                "Tidak Kecanduan": [
                    "✅ Pertahankan screen time saat ini",
                    "✅ Jaga jadwal tidur teratur",
                    "✅ Lanjutkan aktivitas produktif offline",
                ],
                "Kecanduan": [
                    "🚨 Aktifkan Digital Wellbeing / Screen Time limiter",
                    "🚨 Kurangi media sosial 1-2 jam/hari",
                    "🚨 Jadwalkan phone-free hours",
                    "🚨 Gunakan Do Not Disturb saat tidur",
                    "🚨 Ganti aktivitas HP dengan hobi offline",
                ],
            }
            box = "info-box" if label == "Tidak Kecanduan" else "danger-box"
            st.markdown(f'<div class="{box}"><strong>💡 Interpretasi:</strong><br>{interp[label]}</div>',
                        unsafe_allow_html=True)
            st.markdown("#### 📝 Rekomendasi")
            for r in recs[label]:
                st.markdown(f"- {r}")

            # Radar
            st.markdown("#### 📈 Profil Penggunaan")
            cats = ["Screen Time", "Sosial Media", "Gaming", "Kerja/Belajar", "Tidur", "Notifikasi"]
            vals = [screen_time/24, social/24, gaming/24, work/24, sleep/24, min(notif/500, 1.0)]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", fillcolor="rgba(102,126,234,0.2)", line_color="#667eea"))
            fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                showlegend=False, height=400, title="Radar Profil Penggunaan")
            st.plotly_chart(fig_r, use_container_width=True)

    elif mode == "📁 Upload CSV":
        st.markdown("### 📁 Upload CSV untuk Prediksi Batch")
        st.markdown("""
        Kolom: `age`, `gender`, `daily_screen_time_hours`, `social_media_hours`,
        `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`,
        `app_opens_per_day`, `weekend_screen_time`, `stress_level`, `academic_work_impact`
        """)

        uploaded = st.file_uploader("Pilih CSV", type=["csv"])
        if uploaded and models_loaded:
            df = pd.read_csv(uploaded)
            st.markdown(f"**Preview** ({len(df)} baris)")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("🔍 Prediksi Semua", type="primary", use_container_width=True):
                with st.spinner(f"🔄 Memproses {len(df)} baris..."):
                    try:
                        res = predict_batch(df, xgb_model, cnn_model, meta_model,
                                            scaler, col_order, metadata, extractor)
                    except Exception as e:
                        st.error(f"Error: {e}")
                        return

                st.markdown("---")
                st.markdown("### 📊 Hasil Prediksi")
                st.dataframe(res, use_container_width=True)

                counts = res["prediction"].value_counts()
                fig_pie = px.pie(values=counts.values, names=counts.index,
                    color=counts.index,
                    color_discrete_map={"Tidak Kecanduan": "#4CAF50", "Kecanduan": "#F44336"},
                    title="Distribusi Prediksi")
                st.plotly_chart(fig_pie, use_container_width=True)

                c1, c2 = st.columns(2)
                for col, lbl, ico in zip([c1, c2], ["Tidak Kecanduan", "Kecanduan"], ["🟢", "🔴"]):
                    cnt = (res["prediction"] == lbl).sum()
                    with col:
                        st.metric(f"{ico} {lbl}", f"{cnt} ({cnt/len(res)*100:.1f}%)")

                st.download_button("📥 Download CSV", res.to_csv(index=False),
                    "prediction_results.csv", "text/csv", use_container_width=True)


if __name__ == "__main__":
    main()
