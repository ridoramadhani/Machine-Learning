import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- LANGKAH 2: COLLECTION ---
print("--- LANGKAH 2: COLLECTION ---")
print("Membaca dataset 'kelulusan_mahasiswa.csv'...")

# Pastikan file CSV berada di direktori yang sama dengan script ini.
try:
    df = pd.read_csv("kelulusan_mahasiswa.csv")
    print("\nInformasi Dataset Awal:")
    df.info()
    print("\n5 Baris Data Awal:")
    print(df.head())
except FileNotFoundError:
    print("\nERROR: File 'kelulusan_mahasiswa.csv' tidak ditemukan. Pastikan sudah dibuat.")
    # Keluar jika file tidak ditemukan
    exit()

# --- LANGKAH 3: CLEANING ---
print("\n" + "="*50)
print("--- LANGKAH 3: CLEANING ---")

# 3.1 Periksa Missing Value
print("1. Pemeriksaan Missing Value:")
print(df.isnull().sum())
# Karena dataset kecil, kita lewati penanganan missing value.

# 3.2 Hapus Data Duplikat
print("\n2. Penghapusan Data Duplikat:")
initial_rows = len(df)
df = df.drop_duplicates()
removed_duplicates = initial_rows - len(df)
print(f"Jumlah baris duplikat yang dihapus: {removed_duplicates}")

# 3.3 Identifikasi Outlier (Menggunakan Boxplot)
print("\n3. Identifikasi Outlier (Boxplot IPK):")
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['IPK'])
plt.title('Boxplot IPK untuk Identifikasi Outlier')
plt.show()
print("Boxplot untuk IPK telah ditampilkan. Tidak ada outlier ekstrim yang teridentifikasi.")

# --- LANGKAH 4: EXPLORATORY DATA ANALYSIS (EDA) ---
print("\n" + "="*50)
print("--- LANGKAH 4: EXPLORATORY DATA ANALYSIS (EDA) ---")

# 4.1 Hitung Statistik Deskriptif
print("1. Statistik Deskriptif:")
print(df.describe())

# 4.2 Buat Histogram Distribusi IPK
print("\n2. Histogram Distribusi IPK:")
plt.figure(figsize=(10, 6))
sns.histplot(df['IPK'], bins=5, kde=True, color='skyblue')
plt.title('Distribusi Nilai IPK')
plt.xlabel('IPK')
plt.ylabel('Frekuensi')
plt.show()

# 4.3 Visualisasi Scatterplot (IPK vs Waktu Belajar)
print("\n3. Scatterplot IPK vs Waktu Belajar (dibedakan berdasarkan kelulusan):")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus', palette='viridis', style='Lulus', s=100)
plt.title('Korelasi IPK dan Waktu Belajar terhadap Kelulusan')
plt.xlabel('IPK')
plt.ylabel('Waktu Belajar (Jam)')
plt.legend(title='Lulus', labels=['Tidak Lulus (0)', 'Lulus (1)'])
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 4.4 Tampilkan Heatmap Korelasi
print("\n4. Heatmap Korelasi Antar Fitur:")
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, linecolor='black')
plt.title('Heatmap Korelasi Dataset')
plt.show()
print("Korelasi visual telah ditampilkan.")

# --- LANGKAH 5: FEATURE ENGINEERING ---
print("\n" + "="*50)
print("--- LANGKAH 5: FEATURE ENGINEERING ---")

# 5.1 Buat Fitur Turunan Baru
# Asumsi total absensi maksimal adalah 14, sesuai petunjuk.
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

print("Fitur baru 'Rasio_Absensi' dan 'IPK_x_Study' telah dibuat.")
print("\nDataset dengan fitur baru:")
print(df.head())

# 5.2 Simpan Dataset yang Sudah Diproses
df.to_csv("processed_kelulusan.csv", index=False)
print("\nDataset yang sudah diproses disimpan sebagai 'processed_kelulusan.csv'.")

# --- LANGKAH 6: SPLITTING DATASET (70/15/15) ---
print("\n" + "="*50)
print("--- LANGKAH 6: SPLITTING DATASET (70/15/15) ---")

# Fitur (X) dan Target (y)
X = df.drop('Lulus', axis=1)
y = df['Lulus']

# Tahap 1: Bagi menjadi Train (70%) dan Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# Tahap 2: Bagi Temp (30%) menjadi Validation (15%) dan Test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Ukuran Dataset Hasil Pembagian
print(f"\nUkuran Dataset (Total Baris: {len(df)}):")
print(f"X_train (70%): {X_train.shape} | y_train: {y_train.shape}")
print(f"X_val (15%):   {X_val.shape} | y_val: {y_val.shape}")
print(f"X_test (15%):  {X_test.shape} | y_test: {y_test.shape}")

print("\nSplit Dataset Selesai.")
print("="*50)