import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Muat data
df = pd.read_csv("processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Split data 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)
print("Ukuran data setelah split:")
print("X_train:", X_train.shape, "X_val:", X_val.shape, "X_test:", X_test.shape)

# ---
# Langkah 2 — Pipeline & Baseline Random Forest
print("\n--- Langkah 2: Baseline Model ---")
num_cols = X_train.select_dtypes(include="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
], remainder="drop")

rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt",
    class_weight="balanced", random_state=42
)

pipe = Pipeline([("pre", pre), ("clf", rf)])
pipe.fit(X_train, y_train)

y_val_pred = pipe.predict(X_val)
print("Baseline RF — F1-macro(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# ---
# Langkah 3 — Validasi Silang
print("\n--- Langkah 3: Validasi Silang (Cross-Validation) ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
print(f"CV F1-macro (train): {scores.mean():.4f} ± {scores.std():.4f}")

# ---
# Langkah 4 — Tuning Ringkas (GridSearch)
print("\n--- Langkah 4: Tuning Model dengan GridSearchCV ---")
param = {
  "clf__max_depth": [None, 12, 20, 30],
  "clf__min_samples_split": [2, 5, 10]
}

gs = GridSearchCV(pipe, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print("Best params:", gs.best_params_)
best_model = gs.best_estimator_
y_val_best = best_model.predict(X_val)
print("Best RF — F1-macro(val):", f1_score(y_val, y_val_best, average="macro"))

# ---
# Langkah 5 — Evaluasi Akhir (Test Set)
print("\n--- Langkah 5: Evaluasi Akhir pada Test Set ---")
final_model = best_model
y_test_pred = final_model.predict(X_test)
print("F1-macro(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix (test):\n", confusion_matrix(y_test, y_test_pred))

# ROC-AUC dan Kurva
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    try:
        print("ROC-AUC(test):", roc_auc_score(y_test, y_test_proba))
    except Exception as e:
        print(f"ROC-AUC tidak dapat dihitung: {e}")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_test.png", dpi=120)

    # Plot Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(rec, prec, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test Set)")
    plt.tight_layout()
    plt.savefig("pr_test.png", dpi=120)

# ---
# Langkah 6 — Pentingnya Fitur
print("\n--- Langkah 6: Pentingnya Fitur (Feature Importance) ---")
try:
    importances = final_model.named_steps["clf"].feature_importances_
    fn = final_model.named_steps["pre"].get_feature_names_out()
    top = sorted(zip(fn, importances), key=lambda x: x[1], reverse=True)
    print("Top 10 feature importance:")
    for name, val in top[:10]:
        print(f"{name}: {val:.4f}")
except Exception as e:
    print("Feature importance tidak tersedia:", e)

# ---
# Langkah 7 — Simpan Model
print("\n--- Langkah 7: Simpan Model ---")
joblib.dump(final_model, "rf_model.pkl")
print("Model disimpan sebagai rf_model.pkl")

# ---
# Langkah 8 — Cek Inference Lokal
print("\n--- Langkah 8: Cek Inference Lokal ---")
try:
    mdl = joblib.load("rf_model.pkl")
    sample = pd.DataFrame([{
      "IPK": 3.4,
      "Jumlah_Absensi": 4,
      "Waktu_Belajar_Jam": 7,
      "Rasio_Absensi": 4/14,
      "IPK_x_Study": 3.4*7
    }])
    prediksi = int(mdl.predict(sample)[0])
    print("Input sample:", sample.to_dict('records')[0])
    print("Prediksi:", "Lulus" if prediksi == 1 else "Tidak Lulus")
except Exception as e:
    print("Gagal melakukan inferensi lokal:", e)