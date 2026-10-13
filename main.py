# Cell 1: Environment Setup & API Authentication.
import os
os.environ["TABPFN_TOKEN"] = "Your API Token Here" # API token from https://platform.priorlabs.ai/account/api-keys

# Cell 2: Data Acquisition, Audit, and Target Distribution Transparency.

# Install library yang dibutuhkan
!pip install tabpfn catboost lightgbm xgboost imbalanced-learn scikit-learn pandas numpy matplotlib seaborn

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import Library Utama
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
sns.set_theme(style="whitegrid")

# Cell 3: Data Acquisition, Audit, and Target Distribution Transparency.
file_path = '/content/drive/MyDrive/DATA/EKSPERIMEN/STUDENT TabPFN25/data.csv'

try:
    df = pd.read_csv(file_path, sep=';')
except:
    df = pd.read_csv(file_path)

# PERBAIKAN: Menghapus karakter Tab (\t) dan spasi berlebih pada semua nama kolom
df.columns = df.columns.str.replace('\t', '').str.strip()

print(f"Dimensi Dataset Keseluruhan: {df.shape[0]} baris, {df.shape[1]} kolom\n")

# Transparansi Data: Angka Distribusi Kelas Asli
print("--- Distribusi Kelas Target (I0 - Original) ---")
target_counts = df['Target'].value_counts()
target_props = df['Target'].value_counts(normalize=True) * 100
for idx in target_counts.index:
    print(f"{idx:<10}: {target_counts[idx]} observasi ({target_props[idx]:.2f}%)")
print("-" * 45)

# Visualisasi Fase 1
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='Target', order=['Graduate', 'Dropout', 'Enrolled'], palette='viridis')
plt.title('Fase 1: Distribusi Kelas Target (I0 - Original)', fontsize=14)
plt.ylabel('Jumlah Mahasiswa')
plt.xlabel('Status Akademik')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')
plt.show()

# Label Encoding untuk Target
le = LabelEncoder()
df['Target_Encoded'] = le.fit_transform(df['Target'])

# Cell 4: Temporal Feature Splitting (T1, T2, T3) & 80/20 Stratified Split.
# Mendefinisikan Skenario Fitur (Dimensi A)
enrollment_features = [
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification', 'Previous qualification (grade)', # <-- \t sudah dihapus di sini
    'Nacionality', "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", 'Admission grade',
    'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder',
    'Age at enrollment', 'International'
]

sem1_features = [
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)'
]

sem2_features = [
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)'
]

X = df.drop(['Target', 'Target_Encoded'], axis=1)
valid_cols = X.columns.tolist()

features_t1 = [col for col in enrollment_features if col in valid_cols]
features_t2 = features_t1 + [col for col in sem1_features if col in valid_cols]
features_t3 = features_t2 + [col for col in sem2_features if col in valid_cols]

# Transparansi Data: Ketersediaan Informasi
print("--- Rincian Jumlah Fitur Berdasarkan Waktu ---")
print(f"T1 (Enrollment Only)      : {len(features_t1)} fitur")
print(f"T2 (Early Academic)       : {len(features_t2)} fitur (T1 + Sem 1)")
print(f"T3 (Full Info Benchmark)  : {len(features_t3)} fitur (T2 + Sem 2)")
print("-" * 45)

# Visualisasi Fase 2
timing_counts = [len(features_t1), len(features_t2), len(features_t3)]
timing_labels = ['T1: Enrollment Only', 'T2: Early Academic', 'T3: Full Info']

plt.figure(figsize=(8, 4))
sns.barplot(x=timing_labels, y=timing_counts, palette='coolwarm')
plt.title('Fase 2: Jumlah Fitur Berdasarkan Skenario Waktu', fontsize=14)
plt.ylabel('Jumlah Fitur')
for i, v in enumerate(timing_counts):
    plt.text(i, v + 0.5, str(v), ha='center', fontsize=12)
plt.show()

# Pembagian Train 80% dan Untouched Test 20%
y = df['Target_Encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

print("\n--- Transparansi Pemisahan Data (Split) ---")
print(f"Data Training (Development 80%) : {X_train.shape[0]} baris")
print(f"Data Testing  (Untouched 20%)   : {X_test.shape[0]} baris")
print("-" * 45)

# Cell 5: Defining Models & Handling Class Imbalance Strategies.
# Menerapkan Random Oversampling (Hanya pada Data Training)
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

# Konversi kembali ke label asli untuk keperluan print
y_train_labels = pd.Series(y_train).map(dict(enumerate(le.classes_)))
y_train_ros_labels = pd.Series(y_train_ros).map(dict(enumerate(le.classes_)))

# Transparansi Data: Dampak Oversampling
print("--- Dampak Random Oversampling (I2) pada 80% Data Training ---")
print("SEBELUM ROS (I0 / I1):")
print(y_train_labels.value_counts().to_string())
print(f"Total Baris: {len(y_train_labels)}")

print("\nSESUDAH ROS (I2):")
print(y_train_ros_labels.value_counts().to_string())
print(f"Total Baris: {len(y_train_ros_labels)}")
print("-" * 60)

# Visualisasi Fase 3
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=y_train, ax=axes[0], palette='viridis')
axes[0].set_title('Training Set Asli (I0)')
axes[0].set_xticks([0,1,2])
axes[0].set_xticklabels(le.classes_)

sns.countplot(x=y_train_ros, ax=axes[1], palette='viridis')
axes[1].set_title('Training Set setelah ROS (I2)')
axes[1].set_xticks([0,1,2])
axes[1].set_xticklabels(le.classes_)

plt.tight_layout()
plt.show()

# Cell 6: Defining Models & Handling Class Imbalance Strategies.
# Menghitung bobot kelas untuk Skenario I1
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weight_dict = dict(zip(np.unique(y_train), class_weights))

# Fungsi Inisialisasi Model
def get_models(imbalance_strategy='I0'):
    models = {
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
        # PERBAIKAN: Menghapus N_ensemble_configurations
        'TabPFN': TabPFNClassifier(device='cuda')
    }

    # Menerapkan pembobotan kelas (I1) untuk tree-based model
    if imbalance_strategy == 'I1':
        models['XGBoost'] = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', scale_pos_weight=class_weights[1])
        models['LightGBM'] = LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
        models['CatBoost'] = CatBoostClassifier(random_state=42, auto_class_weights='Balanced', verbose=0)
        # TabPFN menangani secara internal atau dapat dibiarkan standar

    return models

def evaluate_model(y_true, y_pred, model_name, scenario_name):
    return {
        'Model': model_name,
        'Scenario': scenario_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Macro_F1': f1_score(y_true, y_pred, average='macro'),
        'Balanced_Acc': balanced_accuracy_score(y_true, y_pred)
    }

# Cell 7: Execution of the 36-Matrix Experiment (5-Fold CV on 80% Development Set).
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import time
import pandas as pd
import numpy as np

# 1. Definisi Kombinasi Skenario Waktu (Dimensi A)
scenarios_time = {
    'T1 (Enrollment-only)': [col for col in features_t1 if col in valid_cols],
    'T2 (Early Academic)': [col for col in features_t2 if col in valid_cols],
    'T3 (Full-info Benchmark)': [col for col in features_t3 if col in valid_cols]
}

# 2. Definisi Kombinasi Imbalance (Dimensi B)
scenarios_imbalance = ['I0', 'I1', 'I2']

# 3. Persiapan K-Fold & ROS
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ros = RandomOverSampler(random_state=42)

results_matrix = []

print("Mulai Eksekusi Pipeline 36 Matriks (Development Set)...\n")

# --- LOOP DIMENSI A (WAKTU) ---
for time_name, time_features in scenarios_time.items():
    print(f"{'='*60}")
    print(f"▶ MENGGUNAKAN WAKTU: {time_name} | Jumlah Fitur: {len(time_features)}")
    print(f"{'='*60}")

    X_time = X_train[time_features]

    # --- LOOP DIMENSI B (IMBALANCE) ---
    for imb_name in scenarios_imbalance:
        print(f"\n  ➤ SKENARIO IMBALANCE: {imb_name}")

        # Penentuan model dasar (I2 menggunakan model I0, I1 menggunakan model I1)
        model_strategy = 'I1' if imb_name == 'I1' else 'I0'
        models = get_models(model_strategy)

        # --- LOOP DIMENSI C (MODEL) ---
        for model_name, model in models.items():
            start_time = time.time()

            fold_metrics = {'macro_f1': [], 'bal_acc': [], 'roc_auc': []}

            # --- 5-FOLD CV PROSES ---
            fold_num = 1
            for train_idx, val_idx in skf.split(X_time, y_train):
                # Memecah data Development menjadi Fold-Train dan Fold-Validation
                X_fold_train, X_fold_val = X_time.iloc[train_idx], X_time.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Menampilkan informasi distribusi data HANYA pada Fold 1 & Model TabPFN agar log tidak penuh
                if fold_num == 1 and model_name == 'TabPFN':
                    if imb_name == 'I0' or imb_name == 'I1':
                        print(f"      [Info Data] Fold-Train: {X_fold_train.shape[0]} baris | Fold-Val: {X_fold_val.shape[0]} baris (Distribusi Asli)")

                # Penerapan Random Oversampling (I2) HANYA pada Fold-Train
                if imb_name == 'I2':
                    X_fold_train, y_fold_train = ros.fit_resample(X_fold_train, y_fold_train)
                    if fold_num == 1 and model_name == 'TabPFN':
                        print(f"      [Info Data] Fold-Train setelah ROS (I2): {X_fold_train.shape[0]} baris | Fold-Val: {X_fold_val.shape[0]} baris")

                # Training dan Prediksi
                model.fit(X_fold_train, y_fold_train)
                preds = model.predict(X_fold_val)
                preds_proba = model.predict_proba(X_fold_val)

                # Kalkulasi Metrik Multiclass
                fold_metrics['macro_f1'].append(f1_score(y_fold_val, preds, average='macro'))
                fold_metrics['bal_acc'].append(balanced_accuracy_score(y_fold_val, preds))
                fold_metrics['roc_auc'].append(roc_auc_score(y_fold_val, preds_proba, multi_class='ovr', average='macro'))

                fold_num += 1

            exec_time = time.time() - start_time

            # Agregasi Rata-rata dari 5 Fold
            mean_macro_f1 = np.mean(fold_metrics['macro_f1'])
            mean_bal_acc = np.mean(fold_metrics['bal_acc'])
            mean_roc_auc = np.mean(fold_metrics['roc_auc'])

            results_matrix.append({
                'Time_Scenario': time_name,
                'Imbalance_Strategy': imb_name,
                'Model': model_name,
                'Mean_Macro_F1': round(mean_macro_f1, 4),
                'Mean_Balanced_Acc': round(mean_bal_acc, 4),
                'Mean_ROC_AUC': round(mean_roc_auc, 4),
                'Exec_Time (s)': round(exec_time, 2)
            })

            print(f"    - {model_name:<10}: Macro F1 = {mean_macro_f1:.4f} | Balanced Acc = {mean_bal_acc:.4f} | Waktu = {exec_time:.1f}s")

# Tampilkan DataFrame Hasil Akhir 36 Matriks
df_final_matrix = pd.DataFrame(results_matrix)
print("\n=== EKSEKUSI SELESAI: HASIL 36 MATRIKS EKSPERIMEN ===")
display(df_final_matrix.sort_values(by=['Time_Scenario', 'Mean_Macro_F1'], ascending=[True, False]).reset_index(drop=True))

# Cell 7B: Robustness Loss Calculation (Delta I0 vs I1/I2).
import pandas as pd

print("=== ANALISIS ROBUSTNESS LOSS (STABILITAS TERHADAP IMBALANCE) ===\n")
print("Rumus: Robustness Loss = Performa(I0) - Performa(I1/I2)")
print("Semakin kecil nilainya (atau negatif), semakin robust model tersebut.\n")

# Membuat tabel pivot dari hasil Sel 7 untuk membandingkan I0 dan I1 pada Macro F1
pivot_robust = df_final_matrix.pivot_table(
    index=['Time_Scenario', 'Model'],
    columns='Imbalance_Strategy',
    values='Mean_Macro_F1'
).reset_index()

# Menghitung Delta (Δ) / Robustness Loss
pivot_robust['Δ (I0 ke I1)'] = pivot_robust['I0'] - pivot_robust['I1']
pivot_robust['Δ (I0 ke I2)'] = pivot_robust['I0'] - pivot_robust['I2']

display(pivot_robust.sort_values(by=['Time_Scenario', 'Δ (I0 ke I1)']))

# Cell 7C: Hyperparameter Tuning (RandomizedSearchCV) for Tree-based models.
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

print("=== HYPERPARAMETER TUNING DENGAN RANDOMIZED SEARCH ===")
print("Catatan: TabPFN menggunakan konfigurasi standar sesuai metodologi.\n")

# Asumsi kita men-tuning pada skenario waktu terbaik (misal: T2)
best_features_tune = [col for col in features_t2 if col in valid_cols]
X_train_tune = X_train[best_features_tune]

# 1. Definisi Ruang Pencarian (Search Space) untuk 3 Model
param_grids = {
    'LightGBM': {
        'model': LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1),
        'params': {
            'num_leaves': [15, 31, 50],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [50, 100, 200]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', scale_pos_weight=weight_dict.get(1, 1)),
        'params': {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [50, 100, 200]
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(random_state=42, auto_class_weights='Balanced', verbose=0),
        'params': {
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [50, 100, 200]
        }
    }
}

best_tuned_models = {}
best_overall_score = 0
best_overall_model_name = ""
final_model = None

# 2. Eksekusi Tuning Bergantian
for model_name, config in param_grids.items():
    print(f"Mencari parameter terbaik untuk {model_name}...")

    random_search = RandomizedSearchCV(
        estimator=config['model'],
        param_distributions=config['params'],
        n_iter=10,        # Mencoba 10 kombinasi acak per model
        scoring='f1_macro',
        cv=3,             # 3-Fold CV untuk tuning
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train_tune, y_train)

    best_tuned_models[model_name] = random_search.best_estimator_
    score = random_search.best_score_

    print(f"  -> Skor Macro F1 Terbaik: {score:.4f}")
    print(f"  -> Parameter: {random_search.best_params_}\n")

    # Menentukan pemenang akhir
    if score > best_overall_score:
        best_overall_score = score
        best_overall_model_name = model_name
        final_model = random_search.best_estimator_

print("="*50)
print(f"PEMENANG TUNING: {best_overall_model_name} dengan Macro F1 = {best_overall_score:.4f}")
print("Model ini akan diteruskan ke Sel 8 untuk Evaluasi Final.")
print("="*50)

# Cell 8: Final Unseen Test Set Evaluation (Classification Report, Confusion Matrix, PR-AUC).
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

print("=== TAHAP FINAL: EVALUASI PADA 20% UNTOUCHED TEST SET ===\n")

# PERBAIKAN: Menggunakan nama variabel yang benar dari Sel 7C (best_features_tune)
X_test_final = X_test[best_features_tune]

# Menggunakan final_model yang sudah di-tuning dari Sel 7C
print(f"Menguji model pemenang ({best_overall_model_name}) pada 20% data testing...\n")
final_preds = final_model.predict(X_test_final)
final_probs = final_model.predict_proba(X_test_final)

# 1. Laporan Klasifikasi
print("--- Laporan Klasifikasi Final ---")
print(classification_report(y_test, final_preds, target_names=le.classes_))

# 2. Visualisasi Confusion Matrix
cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Final Confusion Matrix ({best_overall_model_name})')
plt.ylabel('Aktual (Ground Truth)')
plt.xlabel('Prediksi Model')
plt.show()

# 3. Visualisasi PR-AUC
# Binarize the output classes for PR-Curve
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i, color in zip(range(n_classes), colors):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], final_probs[:, i])
    ap = average_precision_score(y_test_bin[:, i], final_probs[:, i])
    plt.plot(recall, precision, color=color, lw=2, label=f'PR Curve: {le.classes_[i]} (AP = {ap:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (PR-AUC) - {best_overall_model_name}')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()

# Cell 9: Statistical Significance Testing (Friedman Test).
from scipy.stats import friedmanchisquare
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

print("=== UJI STATISTIK FRIEDMAN (Evaluasi Signifikansi 4 Model) ===")

# Fokuskan uji pada Skenario Waktu dan Imbalance terbaik (Contoh: T2 dan I1)
X_stat = X_train[[col for col in features_t2 if col in valid_cols]]
y_stat = y_train
models_stat = get_models('I1')
skf_stat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Dictionary untuk menyimpan daftar skor 5-Fold asli dari masing-masing model
scores_per_model = {name: [] for name in models_stat.keys()}

print("Mengekstraksi skor 5-Fold secara *real-time* untuk tiap model...")
for train_idx, val_idx in skf_stat.split(X_stat, y_stat):
    X_fold_train, X_fold_val = X_stat.iloc[train_idx], X_stat.iloc[val_idx]
    y_fold_train, y_fold_val = y_stat.iloc[train_idx], y_stat.iloc[val_idx]

    for name, model in models_stat.items():
        # Latih dan prediksi cepat
        model.fit(X_fold_train, y_fold_train)
        preds = model.predict(X_fold_val)
        # Tangkap skor asli
        scores_per_model[name].append(f1_score(y_fold_val, preds, average='macro'))

print("\nSkor 5-Fold Asli Berhasil Didapatkan:")
for name, scores in scores_per_model.items():
    print(f"- {name:<10}: {np.round(scores, 4)}")

# Eksekusi Uji Friedman menggunakan array asli
stat, p_value = friedmanchisquare(
    scores_per_model['TabPFN'],
    scores_per_model['XGBoost'],
    scores_per_model['LightGBM'],
    scores_per_model['CatBoost']
)

print(f"\nNilai Statistik Friedman : {stat:.4f}")
print(f"P-Value                  : {p_value:.4f}")

if p_value < 0.05:
    print("Kesimpulan: Terdapat perbedaan performa yang SIGNIFIKAN antar model (Tolak H0).")
else:
    print("Kesimpulan: TIDAK ada perbedaan performa yang signifikan antar model (Terima H0).")

# Cell 10: Model Explainability using SHAP (TreeExplainer/KernelExplainer).
!pip install shap -q
import shap
import matplotlib.pyplot as plt

print("=== ANALISIS FITUR TERPENTING (SHAP) ===")

# Mendeteksi jenis model pemenang
model_name_str = str(type(final_model)).lower()

if "tabpfn" in model_name_str:
    print("Model yang diuji adalah TabPFN (Transformer-based).")
    print("Menggunakan KernelExplainer... (PERINGATAN: Proses ini memakan waktu cukup lama)\n")

    # KernelExplainer sangat berat, jadi kita gunakan sampel kecil dari data latih sebagai background
    background_data = shap.sample(X_train_tune, 50)
    # TabPFN menggunakan .predict_proba untuk memetakan probabilitas multikelas
    explainer = shap.KernelExplainer(final_model.predict_proba, background_data)

    # Menghitung SHAP pada sebagian kecil Test Set agar Google Colab tidak crash/timeout
    test_sample = X_test_final.sample(n=30, random_state=42)
    print("Menghitung SHAP values untuk TabPFN (mohon tunggu)...")
    shap_values = explainer.shap_values(test_sample)

    plt.figure(figsize=(10, 6))
    # SHAP Kernel mereturn list untuk multiclass, summary_plot butuh format spesifik
    shap.summary_plot(shap_values, test_sample, feature_names=X_test_final.columns, class_names=le.classes_, plot_type="bar", show=False)
    plt.title("Tingkat Kepentingan Fitur - TabPFN (KernelExplainer)")
    plt.show()

else:
    print(f"Model yang diuji adalah Tree-based ({best_overall_model_name}).")
    print("Menggunakan TreeExplainer (Optimal & Cepat)...\n")

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_final)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_final, class_names=le.classes_, plot_type="bar", show=False)
    plt.title(f"Tingkat Kepentingan Fitur SHAP - {best_overall_model_name}")
    plt.show()

