"""
================================================================================
TRAINING SCRIPT: Hybrid CNN×XGBoost — Deteksi Kecanduan Smartphone
Target: addicted_label (Binary: 0 = Tidak Kecanduan, 1 = Kecanduan)
================================================================================
Pipeline:
  1. Load & Audit Dataset
  2. Data Cleaning (duplikat, missing, range, outlier)
  3. EDA (Exploratory Data Analysis)
  4. Feature Engineering (15 fitur baru)
  5. Feature Selection (Mutual Information + korelasi tinggi)
  6. Class Imbalance Handling (perbandingan 6 metode)
  7. Train-Test Split & Preprocessing
  8. Optuna Hyperparameter Tuning (XGBoost 100 trials, CNN 50 trials)
  9. Training Final: CNN → Feature Extraction → XGBoost → Meta-Learner
  10. Evaluasi Komprehensif (Accuracy, F1, Kappa, MCC, AUC, ROC)
  11. Cross-Validation 5-Fold Tanpa Data Leakage
  12. SHAP Explainability
  13. Simpan Model

Cara menjalankan:
  python train_model.py

Dependensi:
  pip install pandas numpy scikit-learn xgboost tensorflow optuna
  pip install imbalanced-learn shap matplotlib seaborn scipy
================================================================================
"""

import os, json, warnings, joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, cohen_kappa_score,
    matthews_corrcoef, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, Input, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import shap
from scipy import stats

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Konfigurasi ──────────────────────────────────────────────
TARGET = 'addicted_label'
TARGET_NAMES = ['Tidak Kecanduan', 'Kecanduan']
N_CLASSES = 2

# ── Path: sesuaikan dengan lokasi folder proyek Anda ──
# Gunakan path relatif (otomatis di folder yang sama dengan script)
# Atau ganti dengan path absolut, contoh:
#   SAVE_DIR = r'E:\Python - Project\phone-addiction-detection\Final Proggres\save_model'
#   FIG_DIR  = r'E:\Python - Project\phone-addiction-detection\Final Proggres\figures'
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save_model')
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# =====================================================================
# BAGIAN 1: LOAD & AUDIT DATASET
# =====================================================================
def load_and_audit(filepath):
    print("=" * 65)
    print("  BAGIAN 1: LOAD & AUDIT KUALITAS DATA")
    print("=" * 65)
    df = pd.read_csv(filepath)
    print(f"Shape awal       : {df.shape}")
    print(f"Duplikat         : {df.duplicated().sum()}")

    missing = df.isnull().sum()
    miss_info = pd.DataFrame({'Missing': missing, '%': (missing / len(df) * 100).round(2)})
    print(f"\nMissing Values:")
    has_miss = miss_info[miss_info['Missing'] > 0]
    print(has_miss.to_string() if len(has_miss) > 0 else "  Tidak ada missing values!")

    print(f"\nDistribusi target ({TARGET}):")
    for val, name in zip([0, 1], TARGET_NAMES):
        cnt = (df[TARGET] == val).sum()
        print(f"  {val} ({name:20s}): {cnt:5d} ({cnt / len(df) * 100:.1f}%)")

    return df


# =====================================================================
# BAGIAN 2: DATA CLEANING
# =====================================================================
def clean_data(df):
    print("\n" + "=" * 65)
    print("  BAGIAN 2: DATA CLEANING")
    print("=" * 65)
    df = df.copy()

    # 2.1 Drop kolom tidak relevan
    drop_cols = [c for c in ['transaction_id', 'user_id', 'addiction_level'] if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"Kolom di-drop      : {drop_cols}")

    # 2.2 Hapus duplikat
    n0 = len(df)
    df = df.drop_duplicates()
    print(f"Duplikat dihapus   : {n0 - len(df)} baris")

    # 2.3 Hapus baris tanpa target
    n0 = len(df)
    df = df.dropna(subset=[TARGET])
    print(f"NaN target hapus   : {n0 - len(df)} baris")

    # 2.4 Validasi range
    valid_ranges = {
        'age': (5, 100),
        'daily_screen_time_hours': (0, 24),
        'social_media_hours': (0, 24),
        'gaming_hours': (0, 24),
        'work_study_hours': (0, 24),
        'sleep_hours': (0, 24),
        'notifications_per_day': (0, 5000),
        'app_opens_per_day': (0, 1000),
        'weekend_screen_time': (0, 24),
    }
    n0 = len(df)
    for col, (lo, hi) in valid_ranges.items():
        if col in df.columns:
            df = df[df[col].between(lo, hi)]
    print(f"Out-of-range hapus : {n0 - len(df)} baris")

    # 2.5 Logical consistency
    time_cols = ['social_media_hours', 'gaming_hours', 'work_study_hours']
    if all(c in df.columns for c in time_cols):
        n0 = len(df)
        df = df[df[time_cols].sum(axis=1) <= 24]
        print(f"Inconsistent hapus : {n0 - len(df)} baris")

    # 2.6 Outlier IQR 1%-99%
    outlier_cols = ['daily_screen_time_hours', 'notifications_per_day', 'app_opens_per_day']
    n0 = len(df)
    for col in outlier_cols:
        if col in df.columns:
            q01, q99 = df[col].quantile(0.01), df[col].quantile(0.99)
            df = df[df[col].between(q01, q99)]
    print(f"Outlier hapus      : {n0 - len(df)} baris")

    # 2.7 Imputasi
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.select_dtypes(include='object').columns if c != TARGET]
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    print(f"\nDataset final: {df.shape}")
    for val, name in zip([0, 1], TARGET_NAMES):
        cnt = (df[TARGET] == val).sum()
        print(f"  {val} ({name}): {cnt} ({cnt / len(df) * 100:.1f}%)")

    return df


# =====================================================================
# BAGIAN 3: EDA
# =====================================================================
def run_eda(df):
    print("\n" + "=" * 65)
    print("  BAGIAN 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 65)

    num_feat = ['age', 'daily_screen_time_hours', 'social_media_hours',
                'gaming_hours', 'work_study_hours', 'sleep_hours',
                'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time']
    num_feat = [c for c in num_feat if c in df.columns]

    print("\nStatistik Deskriptif:")
    print(df[num_feat].describe().round(2).to_string())

    # Distribusi per kelas
    colors = {0: '#4CAF50', 1: '#F44336'}
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('Distribusi Fitur per Kelas Kecanduan', fontsize=14, fontweight='bold')
    for i, col in enumerate(num_feat[:9]):
        ax = axes.flatten()[i]
        for val, name in zip([0, 1], TARGET_NAMES):
            subset = df[df[TARGET] == val][col]
            ax.hist(subset, bins=25, alpha=0.5, label=name, color=colors[val])
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig1_distribusi.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Heatmap korelasi
    corr = df[num_feat].corr()
    plt.figure(figsize=(11, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, square=True, linewidths=0.5)
    plt.title('Matriks Korelasi', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig2_korelasi.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Figures saved to {FIG_DIR}/")


# =====================================================================
# BAGIAN 4: FEATURE ENGINEERING & ENCODING
# =====================================================================
def feature_engineering(df):
    print("\n" + "=" * 65)
    print("  BAGIAN 4: FEATURE ENGINEERING & ENCODING")
    print("=" * 65)
    df_fe = df.copy()
    eps = 1e-5

    # Usage Ratios
    df_fe['social_ratio'] = df_fe['social_media_hours'] / (df_fe['daily_screen_time_hours'] + eps)
    df_fe['gaming_ratio'] = df_fe['gaming_hours'] / (df_fe['daily_screen_time_hours'] + eps)
    df_fe['productive_ratio'] = df_fe['work_study_hours'] / (df_fe['daily_screen_time_hours'] + eps)
    df_fe['passive_ratio'] = (df_fe['social_media_hours'] + df_fe['gaming_hours']) / (df_fe['daily_screen_time_hours'] + eps)

    # Wellbeing
    df_fe['sleep_deficit'] = np.maximum(0, 7 - df_fe['sleep_hours'])
    df_fe['sleep_screen_ratio'] = df_fe['sleep_hours'] / (df_fe['daily_screen_time_hours'] + eps)
    df_fe['leisure_total'] = df_fe['social_media_hours'] + df_fe['gaming_hours']

    # Behavioral Intensity
    df_fe['notif_density'] = df_fe['notifications_per_day'] / (df_fe['daily_screen_time_hours'] + eps)
    df_fe['app_open_rate'] = df_fe['app_opens_per_day'] / (df_fe['daily_screen_time_hours'] + eps)
    df_fe['weekend_ratio'] = df_fe['weekend_screen_time'] / (df_fe['daily_screen_time_hours'] + eps)
    df_fe['app_per_notif'] = df_fe['app_opens_per_day'] / (df_fe['notifications_per_day'] + eps)

    # Composite
    df_fe['risk_score'] = (
        df_fe['daily_screen_time_hours'] * 0.3 +
        df_fe['social_media_hours'] * 0.2 +
        df_fe['sleep_deficit'] * 0.25 +
        df_fe['notif_density'] / 10 * 0.15 +
        df_fe['gaming_hours'] * 0.1
    )

    # Binned
    df_fe['screen_time_bin'] = pd.cut(
        df_fe['daily_screen_time_hours'],
        bins=[-0.001, 2, 4, 6, 9, 24], labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(2.0)
    df_fe['sleep_quality'] = pd.cut(
        df_fe['sleep_hours'],
        bins=[-0.001, 5, 6, 7, 8, 24], labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(2.0)

    print(f"Fitur sebelum FE: {df.shape[1]}")
    print(f"Fitur setelah FE: {df_fe.shape[1]}")

    # Target
    y = df_fe[TARGET].astype(int).values

    # Encode kategorik
    X_raw = df_fe.drop(columns=[TARGET])
    cat_features = X_raw.select_dtypes(include='object').columns.tolist()
    print(f"Fitur kategorik : {cat_features}")

    # Mapping manual — konsisten dengan deployment
    if 'gender' in X_raw.columns:
        X_raw['gender'] = X_raw['gender'].map({'Female': 0, 'Male': 1, 'Other': 2}).fillna(2)
    if 'stress_level' in X_raw.columns:
        X_raw['stress_level'] = X_raw['stress_level'].map({'Low': 0, 'Medium': 1, 'High': 2}).fillna(1)
    if 'academic_work_impact' in X_raw.columns:
        X_raw['academic_work_impact'] = X_raw['academic_work_impact'].map({'No': 0, 'Yes': 1}).fillna(0)

    # Handle Inf/NaN
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy='median')
    X_arr = imputer.fit_transform(X_raw)
    X_df = pd.DataFrame(X_arr, columns=X_raw.columns)

    print(f"Shape X: {X_df.shape}, Shape y: {y.shape}")
    return X_df, y, imputer


# =====================================================================
# BAGIAN 5: FEATURE SELECTION
# =====================================================================
def feature_selection(X_df, y):
    print("\n" + "=" * 65)
    print("  BAGIAN 5: FEATURE SELECTION")
    print("=" * 65)

    mi_scores = mutual_info_classif(X_df, y, random_state=SEED)
    mi_df = pd.DataFrame({'feature': X_df.columns, 'mi_score': mi_scores}).sort_values('mi_score', ascending=False)
    print("Top 15 fitur (MI):")
    print(mi_df.head(15).to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    top_n = min(25, len(mi_df))
    mi_top = mi_df.head(top_n)
    ax.barh(mi_top['feature'][::-1], mi_top['mi_score'][::-1],
            color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, top_n)))
    ax.set_xlabel('Mutual Information Score')
    ax.set_title('Feature Importance — Mutual Information', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig3_mutual_info.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Filter MI ≈ 0
    selected = mi_df[mi_df['mi_score'] > 0.001]['feature'].tolist()
    dropped = mi_df[mi_df['mi_score'] <= 0.001]['feature'].tolist()
    if dropped:
        print(f"Fitur dibuang (MI≤0.001): {dropped}")
    X_sel = X_df[selected].copy()

    # Korelasi tinggi
    sc_tmp = StandardScaler()
    corr_mat = pd.DataFrame(sc_tmp.fit_transform(X_sel), columns=X_sel.columns).corr().abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    drop_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
    if drop_corr:
        print(f"Fitur dibuang (corr>0.95): {drop_corr}")

    X_final = X_sel.drop(columns=drop_corr)
    print(f"✅ Fitur final: {X_final.shape[1]}")
    return X_final


# =====================================================================
# BAGIAN 6: CLASS IMBALANCE
# =====================================================================
def compare_imbalance(X_final, y):
    print("\n" + "=" * 65)
    print("  BAGIAN 6: PENANGANAN CLASS IMBALANCE")
    print("=" * 65)

    ir = pd.Series(y).value_counts().max() / pd.Series(y).value_counts().min()
    print(f"Imbalance Ratio: {ir:.2f}")

    scaler = StandardScaler()
    X_cmp = scaler.fit_transform(X_final)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    methods = {
        'No Handling': {'resample': None, 'cw': None},
        'class_weight': {'resample': None, 'cw': 'balanced'},
        'SMOTE': {'resample': SMOTE(random_state=SEED), 'cw': None},
        'ADASYN': {'resample': ADASYN(random_state=SEED), 'cw': None},
        'SMOTETomek': {'resample': SMOTETomek(random_state=SEED), 'cw': None},
        'SMOTE+cw': {'resample': SMOTE(random_state=SEED), 'cw': 'balanced'},
    }

    results = {}
    for name, cfg in methods.items():
        fold_scores = []
        for tr_i, vl_i in skf.split(X_cmp, y):
            X_tr, X_vl = X_cmp[tr_i], X_cmp[vl_i]
            y_tr, y_vl = y[tr_i], y[vl_i]
            if cfg['resample']:
                try:
                    X_tr, y_tr = cfg['resample'].fit_resample(X_tr, y_tr)
                except:
                    pass
            if cfg['cw'] == 'balanced':
                cls = np.unique(y_tr)
                n = len(y_tr)
                wt = {c: n / (len(cls) * (y_tr == c).sum()) for c in cls}
                sw = np.array([wt[yi] for yi in y_tr])
                m = XGBClassifier(n_estimators=100, max_depth=5, eval_metric='logloss',
                                random_state=SEED, verbosity=0)
                m.fit(X_tr, y_tr, sample_weight=sw)
            else:
                m = XGBClassifier(n_estimators=100, max_depth=5, eval_metric='logloss',
                                random_state=SEED, verbosity=0)
                m.fit(X_tr, y_tr)
            fold_scores.append(f1_score(y_vl, m.predict(X_vl), average='binary'))
        results[name] = fold_scores
        print(f"  {name:20s}: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    best = max(results, key=lambda k: np.mean(results[k]))
    print(f"\n✅ Metode terbaik: {best} (F1={np.mean(results[best]):.4f})")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    names_list = list(results.keys())
    means = [np.mean(v) for v in results.values()]
    stds = [np.std(v) for v in results.values()]
    colors = ['#4CAF50' if n == best else '#90A4AE' for n in names_list]
    ax.bar(names_list, means, yerr=stds, capsize=5, color=colors, edgecolor='black')
    ax.set_title('Perbandingan Metode Class Imbalance', fontweight='bold')
    ax.set_ylabel('F1-Score')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig4_imbalance.png', dpi=150, bbox_inches='tight')
    plt.close()
    return best


# =====================================================================
# BAGIAN 7: SPLIT & PREPROCESS
# =====================================================================
def prepare_data(X_final, y, best_method):
    print("\n" + "=" * 65)
    print("  BAGIAN 7: TRAIN-TEST SPLIT & PREPROCESSING")
    print("=" * 65)

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_final.values, y, test_size=0.2, stratify=y, random_state=SEED)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_raw)
    X_te_sc = scaler.transform(X_te_raw)

    # Resample
    if best_method == 'SMOTE':
        X_tr_res, y_tr_res = SMOTE(random_state=SEED).fit_resample(X_tr_sc, y_tr)
        use_sw = False
    elif best_method == 'ADASYN':
        X_tr_res, y_tr_res = ADASYN(random_state=SEED).fit_resample(X_tr_sc, y_tr)
        use_sw = False
    elif best_method == 'SMOTETomek':
        X_tr_res, y_tr_res = SMOTETomek(random_state=SEED).fit_resample(X_tr_sc, y_tr)
        use_sw = False
    elif best_method == 'class_weight':
        X_tr_res, y_tr_res = X_tr_sc, y_tr
        use_sw = True
    elif best_method == 'SMOTE+cw':
        X_tr_res, y_tr_res = SMOTE(random_state=SEED).fit_resample(X_tr_sc, y_tr)
        use_sw = True
    else:
        X_tr_res, y_tr_res = X_tr_sc, y_tr
        use_sw = False

    cls = np.unique(y_tr_res)
    n = len(y_tr_res)
    cw = {c: n / (len(cls) * (y_tr_res == c).sum()) for c in cls}
    sw = np.array([cw[yi] for yi in y_tr_res]) if use_sw else None

    print(f"Train: {X_tr_res.shape}, Test: {X_te_sc.shape}")
    for v, nm in zip([0, 1], TARGET_NAMES):
        cnt = (y_tr_res == v).sum()
        print(f"  {nm}: {cnt} ({cnt / len(y_tr_res) * 100:.1f}%)")

    return X_tr_res, X_te_sc, y_tr_res, y_te, scaler, sw, use_sw


# =====================================================================
# BAGIAN 8: OPTUNA TUNING
# =====================================================================
def build_cnn(trial, input_shape):
    model = Sequential(name="CNN_Extractor")
    model.add(Input(shape=input_shape))

    f1 = trial.suggest_categorical('filters1', [32, 64, 128])
    k1 = trial.suggest_categorical('kernel1', [2, 3, 5])
    model.add(Conv1D(f1, kernel_size=k1, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(trial.suggest_float('drop1', 0.1, 0.4)))

    if trial.suggest_categorical('use_conv2', [True, False]):
        f2 = trial.suggest_categorical('filters2', [64, 128, 256])
        model.add(Conv1D(f2, kernel_size=2, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(trial.suggest_float('drop2', 0.1, 0.4)))

    model.add(GlobalAveragePooling1D())

    d1 = trial.suggest_categorical('dense1', [32, 64, 128])
    model.add(Dense(d1, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(trial.suggest_float('drop_dense', 0.1, 0.4)))

    model.add(Dense(N_CLASSES, activation='softmax'))

    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    model.compile(optimizer=Adam(learning_rate=lr),
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def tune_models(X_tr, y_tr, sw, features):
    print("\n" + "=" * 65)
    print("  BAGIAN 8: HYPERPARAMETER TUNING (OPTUNA)")
    print("=" * 65)

    # Semantic ordering
    semantic = [
        'daily_screen_time_hours', 'weekend_screen_time', 'weekend_ratio', 'screen_time_bin',
        'social_media_hours', 'social_ratio', 'gaming_hours', 'gaming_ratio',
        'leisure_total', 'passive_ratio',
        'work_study_hours', 'productive_ratio',
        'notifications_per_day', 'notif_density',
        'app_opens_per_day', 'app_open_rate', 'app_per_notif',
        'sleep_hours', 'sleep_deficit', 'sleep_screen_ratio', 'sleep_quality',
        'risk_score', 'age',
    ]
    col_ord = [c for c in semantic if c in features]
    col_ord += [c for c in features if c not in col_ord]

    X_tr_df = pd.DataFrame(X_tr, columns=features)
    X_tr_cnn = X_tr_df[col_ord].values.reshape(-1, len(col_ord), 1)

    # XGBoost tuning
    def obj_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 2.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 2.0, log=True),
            'eval_metric': 'logloss', 'random_state': SEED, 'verbosity': 0,
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = []
        for tr_i, vl_i in skf.split(X_tr, y_tr):
            swi = sw[tr_i] if sw is not None else None
            m = XGBClassifier(**params)
            m.fit(X_tr[tr_i], y_tr[tr_i], sample_weight=swi)
            scores.append(f1_score(y_tr[vl_i], m.predict(X_tr[vl_i]), average='binary'))
        return np.mean(scores)

    print("\n🔍 Optuna XGBoost — 100 trials ...")
    study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_xgb.optimize(obj_xgb, n_trials=100, show_progress_bar=True)
    best_xgb = study_xgb.best_params
    print(f"✅ Best XGBoost F1: {study_xgb.best_value:.4f}")

    # CNN tuning
    def obj_cnn(trial):
        model = build_cnn(trial, (len(col_ord), 1))
        cb = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
        hist = model.fit(X_tr_cnn, y_tr, validation_split=0.15, epochs=80,
                        batch_size=trial.suggest_categorical('batch_size', [32, 64, 128]),
                        callbacks=[cb], verbose=0)
        tf.keras.backend.clear_session()
        return max(hist.history['val_accuracy'])

    print("\n🔍 Optuna CNN — 50 trials ...")
    study_cnn = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_cnn.optimize(obj_cnn, n_trials=50, show_progress_bar=True)
    print(f"✅ Best CNN Acc: {study_cnn.best_value:.4f}")

    return best_xgb, study_cnn, col_ord


# =====================================================================
# BAGIAN 9: TRAINING FINAL
# =====================================================================
def train_final(X_tr, X_te, y_tr, y_te, sw, best_xgb, study_cnn, col_ord, features):
    print("\n" + "=" * 65)
    print("  BAGIAN 9: TRAINING MODEL FINAL")
    print("=" * 65)

    X_tr_df = pd.DataFrame(X_tr, columns=features)
    X_te_df = pd.DataFrame(X_te, columns=features)
    X_tr_cnn = X_tr_df[col_ord].values.reshape(-1, len(col_ord), 1)
    X_te_cnn = X_te_df[col_ord].values.reshape(-1, len(col_ord), 1)

    # CNN
    print("\n📌 Training CNN ...")
    cnn_model = build_cnn(study_cnn.best_trial, (len(col_ord), 1))
    cnn_model.summary()
    cbs = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7)]
    history = cnn_model.fit(X_tr_cnn, y_tr, validation_split=0.15, epochs=150,
                            batch_size=study_cnn.best_params.get('batch_size', 64),
                            callbacks=cbs, verbose=1)

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, m, t in zip(axes, ['accuracy', 'loss'], ['Accuracy', 'Loss']):
        ax.plot(history.history[m], label='Train', color='#2196F3')
        ax.plot(history.history[f'val_{m}'], label='Val', color='#FF5722')
        ax.set_title(t); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig5_cnn_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Extract CNN features
    print("\n📌 Extracting CNN features ...")
    extractor = Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[-3].output)
    cnn_feat_tr = extractor.predict(X_tr_cnn, verbose=0)
    cnn_feat_te = extractor.predict(X_te_cnn, verbose=0)
    X_tr_aug = np.hstack([X_tr, cnn_feat_tr])
    X_te_aug = np.hstack([X_te, cnn_feat_te])
    print(f"Augmented shape: {X_tr_aug.shape}")

    # XGBoost
    print("\n📌 Training XGBoost (augmented) ...")
    best_xgb.update({'eval_metric': 'logloss', 'random_state': SEED, 'verbosity': 0})
    xgb_model = XGBClassifier(**best_xgb)
    xgb_model.fit(X_tr_aug, y_tr, sample_weight=sw, eval_set=[(X_te_aug, y_te)], verbose=False)

    # Baseline
    xgb_base = XGBClassifier(**best_xgb)
    xgb_base.fit(X_tr, y_tr, sample_weight=sw, verbose=False)

    # Meta-learner
    print("\n📌 Training Meta-Learner (Stacking) ...")
    xgb_tr_prob = xgb_model.predict_proba(X_tr_aug)
    cnn_tr_prob = cnn_model.predict(X_tr_cnn, verbose=0)
    stack_tr = np.hstack([xgb_tr_prob, cnn_tr_prob])

    xgb_te_prob = xgb_model.predict_proba(X_te_aug)
    cnn_te_prob = cnn_model.predict(X_te_cnn, verbose=0)
    stack_te = np.hstack([xgb_te_prob, cnn_te_prob])

    meta_cv = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=SEED),
        {'C': [0.001, 0.01, 0.1, 1, 10, 100]}, cv=5, scoring='f1')
    meta_cv.fit(stack_tr, y_tr)
    meta_model = meta_cv.best_estimator_
    print(f"✅ Meta-learner: C={meta_cv.best_params_['C']}, CV F1={meta_cv.best_score_:.4f}")

    return (cnn_model, xgb_model, xgb_base, meta_model, extractor,
            X_tr_aug, X_te_aug, X_tr_cnn, X_te_cnn,
            cnn_feat_te, xgb_te_prob, cnn_te_prob, stack_te)


# =====================================================================
# BAGIAN 10: EVALUASI
# =====================================================================
def eval_model(name, y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    kap = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob[:, 1]) if y_prob is not None else None

    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"{'=' * 55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Kappa    : {kap:.4f}")
    print(f"  MCC      : {mcc:.4f}")
    if auc: print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES))
    return {'model': name, 'accuracy': acc, 'f1': f1, 'kappa': kap, 'mcc': mcc, 'auc': auc}


def full_evaluation(y_te, cnn_model, xgb_model, xgb_base, meta_model,
                    X_te, X_te_aug, X_te_cnn, xgb_te_prob, cnn_te_prob, stack_te):
    print("\n" + "=" * 65)
    print("  BAGIAN 10: EVALUASI KOMPREHENSIF")
    print("=" * 65)

    cnn_pred = np.argmax(cnn_te_prob, axis=1)
    xgb_b_pred = xgb_base.predict(X_te)
    xgb_b_prob = xgb_base.predict_proba(X_te)
    xgb_pred = xgb_model.predict(X_te_aug)
    hyb_pred = meta_model.predict(stack_te)
    hyb_prob = meta_model.predict_proba(stack_te)

    results = []
    results.append(eval_model('XGBoost Baseline', y_te, xgb_b_pred, xgb_b_prob))
    results.append(eval_model('CNN Extractor', y_te, cnn_pred, cnn_te_prob))
    results.append(eval_model('XGBoost+CNN', y_te, xgb_pred, xgb_te_prob))
    results.append(eval_model('Hybrid Stacking', y_te, hyb_pred, hyb_prob))

    rdf = pd.DataFrame(results).set_index('model')
    print("\n" + rdf.round(4).to_string())

    # Confusion Matrix
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle('Confusion Matrix', fontsize=14, fontweight='bold')
    for ax, (n, p, cm) in zip(axes, [
        ('XGB Baseline', xgb_b_pred, 'Blues'),
        ('CNN', cnn_pred, 'Greens'),
        ('XGB+CNN', xgb_pred, 'Oranges'),
        ('Hybrid', hyb_pred, 'Purples')]):
        ConfusionMatrixDisplay(confusion_matrix(y_te, p), display_labels=TARGET_NAMES).plot(ax=ax, cmap=cm, colorbar=False)
        ax.set_title(n)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig6_confusion.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    for n, prob, c in [('XGB Baseline', xgb_b_prob, '#2196F3'),
                        ('CNN', cnn_te_prob, '#4CAF50'),
                        ('XGB+CNN', xgb_te_prob, '#FF9800'),
                        ('Hybrid', hyb_prob, '#9C27B0')]:
        fpr, tpr, _ = roc_curve(y_te, prob[:, 1])
        auc_val = roc_auc_score(y_te, prob[:, 1])
        ax.plot(fpr, tpr, color=c, label=f'{n} (AUC={auc_val:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curve', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig7_roc.png', dpi=150, bbox_inches='tight')
    plt.close()

    return rdf


# =====================================================================
# BAGIAN 11: CV TANPA LEAKAGE
# =====================================================================
def cv_no_leakage(X_final, y, best_xgb, study_cnn, col_ord, best_method, meta_C):
    print("\n" + "=" * 65)
    print("  BAGIAN 11: 5-FOLD CV (BEBAS LEAKAGE)")
    print("=" * 65)

    X_cv = X_final.values
    feats = list(X_final.columns)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    cv_res = {'XGB Baseline': [], 'CNN': [], 'XGB+CNN': [], 'Hybrid': []}

    for fold, (tr_i, vl_i) in enumerate(skf.split(X_cv, y)):
        print(f"\n  ── Fold {fold + 1}/5 ──")
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_cv[tr_i])
        X_vl = sc.transform(X_cv[vl_i])
        y_tr, y_vl = y[tr_i], y[vl_i]

        # Resample
        try:
            if best_method in ('SMOTE', 'SMOTE+cw'):
                X_tr, y_tr = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
            elif best_method == 'ADASYN':
                X_tr, y_tr = ADASYN(random_state=SEED).fit_resample(X_tr, y_tr)
            elif best_method == 'SMOTETomek':
                X_tr, y_tr = SMOTETomek(random_state=SEED).fit_resample(X_tr, y_tr)
        except:
            pass

        use_sw = best_method in ('class_weight', 'SMOTE+cw')
        sw_f = None
        if use_sw:
            cls = np.unique(y_tr)
            n = len(y_tr)
            cw = {c: n / (len(cls) * (y_tr == c).sum()) for c in cls}
            sw_f = np.array([cw[yi] for yi in y_tr])

        col_f = [c for c in col_ord if c in feats]
        col_f += [c for c in feats if c not in col_f]
        X_tr_cnn = pd.DataFrame(X_tr, columns=feats)[col_f].values.reshape(-1, len(col_f), 1)
        X_vl_cnn = pd.DataFrame(X_vl, columns=feats)[col_f].values.reshape(-1, len(col_f), 1)

        # XGB Baseline
        xb = XGBClassifier(**best_xgb); xb.fit(X_tr, y_tr, sample_weight=sw_f)
        cv_res['XGB Baseline'].append(f1_score(y_vl, xb.predict(X_vl), average='binary'))

        # CNN
        cn = build_cnn(study_cnn.best_trial, (len(col_f), 1))
        cn.fit(X_tr_cnn, y_tr, validation_split=0.1, epochs=60, batch_size=64,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], verbose=0)
        cp = cn.predict(X_vl_cnn, verbose=0)
        cv_res['CNN'].append(f1_score(y_vl, np.argmax(cp, axis=1), average='binary'))

        # XGB+CNN
        ext = Model(inputs=cn.input, outputs=cn.layers[-3].output)
        X_tr_a = np.hstack([X_tr, ext.predict(X_tr_cnn, verbose=0)])
        X_vl_a = np.hstack([X_vl, ext.predict(X_vl_cnn, verbose=0)])
        xa = XGBClassifier(**best_xgb); xa.fit(X_tr_a, y_tr, sample_weight=sw_f)
        cv_res['XGB+CNN'].append(f1_score(y_vl, xa.predict(X_vl_a), average='binary'))

        # Hybrid
        st_tr = np.hstack([xa.predict_proba(X_tr_a), cn.predict(X_tr_cnn, verbose=0)])
        st_vl = np.hstack([xa.predict_proba(X_vl_a), cp])
        mf = LogisticRegression(C=meta_C, max_iter=2000); mf.fit(st_tr, y_tr)
        cv_res['Hybrid'].append(f1_score(y_vl, mf.predict(st_vl), average='binary'))

        print(f"    XGB={cv_res['XGB Baseline'][-1]:.4f}  CNN={cv_res['CNN'][-1]:.4f}  "
            f"XGB+CNN={cv_res['XGB+CNN'][-1]:.4f}  Hybrid={cv_res['Hybrid'][-1]:.4f}")
        tf.keras.backend.clear_session()

    print("\n" + "=" * 55)
    for k, v in cv_res.items():
        print(f"  {k:20s}: {np.mean(v):.4f} ± {np.std(v):.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    cv_df = pd.DataFrame(cv_res)
    bplot = ax.boxplot(cv_df.values, patch_artist=True)
    for p, c in zip(bplot['boxes'], ['#90A4AE', '#4CAF50', '#2196F3', '#9C27B0']):
        p.set_facecolor(c); p.set_alpha(0.6)
    ax.set_xticklabels(cv_df.columns, rotation=15)
    ax.set_ylabel('F1-Score'); ax.set_title('5-Fold CV', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig8_cv.png', dpi=150, bbox_inches='tight')
    plt.close()


# =====================================================================
# BAGIAN 12: SHAP
# =====================================================================
def shap_analysis(xgb_model, X_te_aug, features, n_cnn):
    print("\n" + "=" * 65)
    print("  BAGIAN 12: SHAP EXPLAINABILITY")
    print("=" * 65)
    aug_names = list(features) + [f'cnn_{i}' for i in range(n_cnn)]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_te_aug)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_te_aug, feature_names=aug_names, show=False, plot_type='bar')
    plt.title('SHAP Feature Importance', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig9_shap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved {FIG_DIR}/fig9_shap.png")


# =====================================================================
# BAGIAN 13: SAVE
# =====================================================================
def save_all(xgb_model, xgb_base, meta_model, scaler, imputer, cnn_model, col_ord, features, best_method):
    print("\n" + "=" * 65)
    print("  BAGIAN 13: SIMPAN MODEL")
    print("=" * 65)

    joblib.dump(xgb_model, f'{SAVE_DIR}/xgb_cnn_augmented.pkl')
    joblib.dump(xgb_base, f'{SAVE_DIR}/xgb_baseline.pkl')
    joblib.dump(meta_model, f'{SAVE_DIR}/meta_model.pkl')
    joblib.dump(scaler, f'{SAVE_DIR}/scaler.pkl')
    joblib.dump(imputer, f'{SAVE_DIR}/imputer.pkl')
    joblib.dump(col_ord, f'{SAVE_DIR}/col_order.pkl')
    cnn_model.save(f'{SAVE_DIR}/cnn_extractor.keras')

    meta = {
        'features': list(features),
        'col_order': col_ord,
        'target': TARGET,
        'target_names': TARGET_NAMES,
        'n_classes': N_CLASSES,
        'best_method': best_method,
    }
    with open(f'{SAVE_DIR}/metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print("✅ Model disimpan di saved_models/")


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("\n" + "█" * 65)
    print("  HYBRID CNN × XGBOOST — DETEKSI KECANDUAN SMARTPHONE")
    print("  Target: addicted_label (Binary Classification)")
    print("█" * 65)

    # Cari file CSV di beberapa lokasi umum
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(script_dir, 'Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv'),
        os.path.join(script_dir, '..', 'Data', 'Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv'),
        os.path.join(script_dir, '..', 'Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv'),
        'Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv',
        'Data/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv',
        '../Data/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv',
    ]
    fp = None
    for p in paths:
        if os.path.exists(p):
            fp = p; break
    if not fp:
        fp = input("Path ke CSV: ").strip()

    df_raw = load_and_audit(fp)
    df = clean_data(df_raw)
    run_eda(df)
    X_df, y, imputer = feature_engineering(df)
    X_final = feature_selection(X_df, y)
    best_method = compare_imbalance(X_final, y)

    feats = list(X_final.columns)
    X_tr, X_te, y_tr, y_te, scaler, sw, use_sw = prepare_data(X_final, y, best_method)
    best_xgb, study_cnn, col_ord = tune_models(X_tr, y_tr, sw, feats)

    (cnn_model, xgb_model, xgb_base, meta_model, extractor,
    X_tr_aug, X_te_aug, X_tr_cnn, X_te_cnn,
    cnn_feat_te, xgb_te_prob, cnn_te_prob, stack_te) = \
        train_final(X_tr, X_te, y_tr, y_te, sw, best_xgb, study_cnn, col_ord, feats)

    rdf = full_evaluation(y_te, cnn_model, xgb_model, xgb_base, meta_model,
                        X_te, X_te_aug, X_te_cnn, xgb_te_prob, cnn_te_prob, stack_te)

    meta_C = meta_model.C if hasattr(meta_model, 'C') else 1.0
    cv_no_leakage(X_final, y, best_xgb, study_cnn, col_ord, best_method, meta_C)

    shap_analysis(xgb_model, X_te_aug, feats, cnn_feat_te.shape[1])
    save_all(xgb_model, xgb_base, meta_model, scaler, imputer, cnn_model, col_ord, feats, best_method)

    best_m = rdf['accuracy'].idxmax()
    print("\n" + "█" * 65)
    print(f"  Best: {best_m}")
    print(f"  Accuracy: {rdf.loc[best_m, 'accuracy']:.4f}  |  F1: {rdf.loc[best_m, 'f1']:.4f}  |  AUC: {rdf.loc[best_m, 'auc']:.4f}")
    print(f"  Selanjutnya: streamlit run streamlit_app.py")
    print("█" * 65)


if __name__ == '__main__':
    main()
