import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Menggunakan try-except untuk penanganan file
try:
    df = pd.read_csv("processed_kelulusan.csv")
except FileNotFoundError:
    print("ERROR: File 'processed_kelulusan.csv' tidak ditemukan. Pastikan file ada di direktori yang sama.")
    # Membuat data dummy agar script tetap berjalan untuk demonstrasi,
    # namun Anda harus menggantinya dengan data nyata.
    print("Menggunakan data dummy untuk melanjutkan demonstrasi...")
    data = {
        'IPK_Semester_1': np.random.uniform(2.5, 4.0, 1000),
        'SKS_Lulus': np.random.randint(0, 144, 1000),
        'Usia_Masuk': np.random.randint(18, 25, 1000),
        'Biaya_Pendidikan': np.random.randint(1000000, 10000000, 1000),
        'Lulus': np.random.randint(0, 2, 1000)
    }
    df = pd.DataFrame(data)


X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Split data menjadi Train, Validation, dan Test (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("="*50)
print("1. DATA SPLITTING")
print(f"Shape X_train: {X_train.shape}, Shape X_val: {X_val.shape}, Shape X_test: {X_test.shape}")


# ==============================================================================
# LANGKAH 2: BASELINE MODEL (LOGISTIC REGRESSION) & PIPELINE
# ==============================================================================

num_cols = X_train.select_dtypes(include="number").columns

# Preprocessing Pipeline (hanya untuk kolom numerik, karena semua kolom di dummy adalah numerik)
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
], remainder="drop")

# Model Baseline
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
pipe_lr = Pipeline([("pre", pre), ("clf", logreg)])

pipe_lr.fit(X_train, y_train)
y_val_pred_lr = pipe_lr.predict(X_val)

print("="*50)
print("2. BASELINE MODEL (LOGISTIC REGRESSION)")
print("Baseline (LogReg) F1(val) Macro:", f1_score(y_val, y_val_pred_lr, average="macro"))
print(classification_report(y_val, y_val_pred_lr, digits=3))


# ==============================================================================
# LANGKAH 3: MODEL ALTERNATIF (RANDOM FOREST)
# ==============================================================================

rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42
)
pipe_rf = Pipeline([("pre", pre), ("clf", rf)])

pipe_rf.fit(X_train, y_train)
y_val_rf = pipe_rf.predict(X_val)

print("="*50)
print("3. MODEL ALTERNATIF (RANDOM FOREST)")
print("RandomForest F1(val) Macro:", f1_score(y_val, y_val_rf, average="macro"))


# ==============================================================================
# LANGKAH 4: VALIDASI SILANG & TUNING RINGKAS (RANDOM FOREST)
# ==============================================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param = {
    "clf__max_depth": [None, 12, 20, 30],
    "clf__min_samples_split": [2, 5, 10]
}
# Menggunakan pipe_rf sebagai estimator
gs = GridSearchCV(pipe_rf, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)

print("="*50)
print("4. VALIDASI SILANG & TUNING RINGKAS (Random Forest)")
print("Memulai GridSearchCV...")

gs.fit(X_train, y_train)

print(f"Best params: {gs.best_params_}")
print(f"Best CV F1 Macro: {gs.best_score_:.4f}")

best_rf = gs.best_estimator_
y_val_best = best_rf.predict(X_val)
print(f"Best RF F1(val) Macro (setelah tuning): {f1_score(y_val, y_val_best, average='macro'):.4f}")


# ==============================================================================
# LANGKAH 5: EVALUASI AKHIR (TEST SET)
# ==============================================================================

# Pemilihan model final (Diasumsikan Random Forest terbaik setelah tuning)
final_model = best_rf

y_test_pred = final_model.predict(X_test)

print("="*50)
print("5. EVALUASI AKHIR PADA TEST SET")
print(f"Model Final: Random Forest (Tuned)")
print(f"F1 Macro (Test Set): {f1_score(y_test, y_test_pred, average='macro'):.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, digits=3))
print("\nConfusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_test_pred))

# ROC-AUC dan Plot
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:,1]
    try:
        roc_auc = roc_auc_score(y_test, y_test_proba)
        print(f"\nROC-AUC (Test Set): {roc_auc:.4f}")
    except ValueError as e:
        print(f"\nTidak dapat menghitung ROC-AUC: {e}. Kemungkinan hanya satu kelas yang ada di y_test.")

    try:
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # garis diagonal
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Curve (Test Set)")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("roc_test.png", dpi=120)
        print("Plot ROC Curve tersimpan sebagai roc_test.png")
    except Exception as e:
        print(f"Gagal membuat plot ROC: {e}")


# ==============================================================================
# LANGKAH 6 (OPSIONAL): SIMPAN MODEL
# ==============================================================================
joblib.dump(final_model, "model.pkl")
print("="*50)
print("6. MODEL SAVING")
print("Model tersimpan ke model.pkl")


# ==============================================================================
# LANGKAH 7 (OPSIONAL): ENDPOINT INFERENCE (FLASK)
# ==============================================================================
from flask import Flask, request, jsonify

app = Flask(__name__)
# Memastikan MODEL dapat dimuat, jika tidak ada, akan error.
# Asumsi model.pkl sudah dibuat di Langkah 6
try:
    MODEL = joblib.load("model.pkl")
except FileNotFoundError:
    print("\n[FLASK] ERROR: model.pkl belum ada. Jalankan Langkah 1-6 terlebih dahulu.")
    MODEL = None # Set None jika gagal load

@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model belum dimuat. Jalankan skrip ini secara lengkap terlebih dahulu."}), 503

    try:
        # data = {"IPK_Semester_1": 3.8, "SKS_Lulus": 130, "Usia_Masuk": 18, "Biaya_Pendidikan": 5000000}
        data = request.get_json(force=True)  # dict fitur
        X_infer = pd.DataFrame([data])
        
        yhat = MODEL.predict(X_infer)[0]
        proba = None
        
        if hasattr(MODEL, "predict_proba"):
            # Ambil probabilitas kelas positif (index 1)
            proba = float(MODEL.predict_proba(X_infer)[:,1][0])
            
        return jsonify({"prediction": int(yhat), "proba": proba})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Uncomment baris di bawah untuk menjalankan Flask server
# if __name__ == "__main__":
#     # PENTING: Untuk menjalankan aplikasi Flask, jalankan file ini dari terminal
#     # dan akses endpoint http://127.0.0.1:5000/predict dengan metode POST.
#     print("="*50)
#     print("7. FLASK INFERENCE ENDPOINT (BELUM DIJALANKAN)")
#     print("Uncomment bagian 'if __name__ == \"__main__\":' untuk menjalankan server.")
#     # app.run(port=5000, debug=False) # Hapus debug=False jika Anda ingin melihat log