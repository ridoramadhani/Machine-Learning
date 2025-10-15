import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Set seed global untuk reproduktifitas
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

print(f"Seed acak global telah disetel ke: {SEED}")
print("-" * 60)

# Langkah 1 — Siapkan Data 

# Muat data
try:
    df = pd.read_csv("processed_kelulusan.csv")
    print(f"Data dimuat. Jumlah baris: {len(df)}")
except FileNotFoundError:
    print("⚠️ ERROR: File 'processed_kelulusan.csv' tidak ditemukan.")
    # Membuat data dummy untuk memungkinkan eksekusi (Ganti dengan data asli Anda)
    print("Menggunakan data dummy untuk melanjutkan kode.")
    data_dummy = np.random.rand(1000, 15)
    labels_dummy = np.random.randint(0, 2, 1000)
    df = pd.DataFrame(data_dummy, columns=[f'feature_{i}' for i in range(15)])
    df['Lulus'] = labels_dummy

X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Standardisasi fitur
sc = StandardScaler()
Xs = sc.fit_transform(X)

# Pembagian data: Train (70%), Val (15%), Test (15%)
# Test size 0.3 dari total, Val size 0.5 dari 0.3 (yaitu 0.15)
X_train, X_temp, y_train, y_temp = train_test_split(
    Xs, y, test_size=0.3, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)

print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}, Test Shape: {X_test.shape}")
print("-" * 60)

# Langkah 2 — Bangun Model ANN (Arsitektur Baseline) 

model = keras.Sequential([
    # Input Layer (implisit: shape ditentukan oleh input pertama)
    layers.Input(shape=(X_train.shape[1],)),
    
    # Hidden Layer 1: 32 neuron, ReLU
    layers.Dense(32, activation="relu"),
    
    # Regularisasi: Dropout 30%
    layers.Dropout(0.3),
    
    # Hidden Layer 2: 16 neuron, ReLU
    layers.Dense(16, activation="relu"),
    
    # Output Layer: 1 neuron, Sigmoid (Klasifikasi Biner)
    layers.Dense(1, activation="sigmoid")
])

# Kompilasi Model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy","AUC"])

print("Arsitektur Model (Baseline):")
model.summary()
print("-" * 60)

# Langkah 3 — Training dengan Early Stopping 

# Early Stopping callback
es = keras.callbacks.EarlyStopping(
    monitor="val_loss",           # Metrik yang dipantau
    patience=10,                  # Jumlah epoch tanpa peningkatan sebelum berhenti
    restore_best_weights=True     # Kembalikan bobot model terbaik
)

print("Memulai Training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,                   # Batas maksimum epoch
    batch_size=32,
    callbacks=[es],
    verbose=1                     # Tampilkan progress
)

print(f"\nTraining selesai. Dihentikan pada epoch ke-{len(history.history['loss'])}")
print("-" * 60)

# Langkah 4 — Evaluasi di Test Set 

# Evaluasi metrik dasar
loss, acc, auc_keras = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Acc: {acc:.4f}")
print(f"Test AUC (dari Keras): {auc_keras:.4f}")

# Prediksi probabilitas dan konversi ke kelas biner (Threshold 0.5)
y_proba = model.predict(X_test, verbose=0).ravel()
y_pred = (y_proba >= 0.5).astype(int)

# Hitung AUC ROC manual (sebagai cross-check)
auc_roc_manual = roc_auc_score(y_test, y_proba)
print(f"Test ROC-AUC (Manual): {auc_roc_manual:.4f}")

# Laporan Klasifikasi
print("\nConfusion Matrix (Threshold 0.5):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report (Termasuk F1-score):")
print(classification_report(y_test, y_pred, digits=3))
print("-" * 60)

# Langkah 5 — Visualisasi Learning Curve

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve: Train vs. Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=120)
plt.show()

print("Learning curve telah disimpan sebagai 'learning_curve.png'.")
print("-" * 60)