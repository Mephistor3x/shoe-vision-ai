import os
import zipfile
import cv2
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
import splitfolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# KONFIGURASI PATH (Ubah sesuai lokasi Anda)
# ==========================================
ZIP_FILE_PATH = 'projek (2).zip'  # Nama file ZIP Anda
EXTRACT_PATH = 'dataset_original'      # Folder hasil ekstrak
CLEAN_DIR = 'dataset_clean'            # Folder hasil pembersihan
FINAL_DIR = 'dataset_final'            # Folder siap training (split)
METADATA_FILE = 'metadata_sepatu.csv'

# ==========================================
# FUNGSI PEMBERSIHAN
# ==========================================

def check_blur(image_path, threshold=100):
    """
    Mengecek apakah gambar terlalu buram untuk dipakai sebagai dataset.

    Parameter:
    - image_path: lokasi file gambar yang akan dicek.
    - threshold: batas minimum nilai ketajaman gambar.

    Return:
    - Tuple (is_blur, score), yaitu status buram dan skor ketajaman.
    """
    image = cv2.imread(image_path)
    if image is None: return True, 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Variance dari Laplacian sering dipakai untuk mengukur ketajaman gambar.
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold, fm

def is_logo_or_banner(width, height):
    """
    Menentukan apakah gambar kemungkinan logo/banner dari perbandingan ukuran.

    Gambar yang terlalu lebar atau terlalu tinggi biasanya bukan foto produk.
    Return True jika rasio gambarnya mencurigakan sebagai logo/banner.
    """
    aspect_ratio = width / height
    if aspect_ratio > 2.5 or aspect_ratio < 0.4:
        return True
    return False

def get_image_hash(image_path):
    """
    Membuat hash sederhana dari isi gambar untuk mendeteksi duplikat.

    Return string hash jika gambar berhasil dibuka, atau None jika file rusak
    atau formatnya tidak bisa dibaca PIL.
    """
    try:
        with Image.open(image_path) as img:
            return str(imagehash.average_hash(img))
    except:
        return None

# ==========================================
# LANGKAH 1: EKSTRAK FILE
# ==========================================
if os.path.exists(ZIP_FILE_PATH):
    print(f"[+] Mengekstrak {ZIP_FILE_PATH}...")
    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
else:
    print(f"[!] File {ZIP_FILE_PATH} tidak ditemukan. Pastikan file ada di folder yang sama.")

# ==========================================
# LANGKAH 2: PROSES PEMBERSIHAN
# ==========================================
print("[+] Memulai proses pembersihan...")

valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
metadata = []
hashes = {}
os.makedirs(CLEAN_DIR, exist_ok=True)

# List semua file gambar
all_files = []
for root, dirs, files in os.walk(EXTRACT_PATH):
    for file in files:
        if file.lower().endswith(valid_extensions):
            all_files.append(os.path.join(root, file))

for img_path in all_files:
    filename = os.path.basename(img_path)
    # Label diambil dari folder induk
    label = os.path.basename(os.path.dirname(img_path))
    
    # Jika label adalah root extract, tandai sebagai 'unlabeled'
    if label == EXTRACT_PATH:
        label = 'unlabeled'

    status_valid = True
    alasan = ""
    
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            fmt = img.format
            
            if width < 200 or height < 200:
                status_valid = False
                alasan = "Resolusi terlalu kecil"
            elif is_logo_or_banner(width, height):
                status_valid = False
                alasan = "Terdeteksi Logo/Banner"
            else:
                buram, score = check_blur(img_path)
                if buram:
                    status_valid = False
                    alasan = f"Buram (Score: {score:.2f})"
                else:
                    h = get_image_hash(img_path)
                    if h in hashes:
                        status_valid = False
                        alasan = f"Duplikat dari {hashes[h]}"
                    else:
                        hashes[h] = filename
    except Exception as e:
        status_valid = False
        alasan = f"File Rusak: {str(e)}"
        width, height, fmt = 0, 0, "Unknown"

    metadata.append({
        'nama_file': filename,
        'path_gambar': img_path,
        'label': label,
        'lebar': width,
        'tinggi': height,
        'format': fmt,
        'status_valid': status_valid,
        'alasan_dibuang': alasan
    })

    if status_valid:
        target_folder = os.path.join(CLEAN_DIR, label)
        os.makedirs(target_folder, exist_ok=True)
        shutil.copy2(img_path, os.path.join(target_folder, filename))

# Simpan Metadata
df_meta = pd.DataFrame(metadata)
df_meta.to_csv(METADATA_FILE, index=False)

# ==========================================
# LANGKAH 3: SPLIT DATASET (70:20:10)
# ==========================================
print("[+] Membagi dataset menjadi Train, Val, dan Test...")
if os.path.exists(CLEAN_DIR) and len(os.listdir(CLEAN_DIR)) > 0:
    splitfolders.ratio(CLEAN_DIR, output=FINAL_DIR, seed=42, ratio=(.7, .2, .1))
else:
    print("[!] Tidak ada gambar valid untuk di-split.")

# ==========================================
# LANGKAH 4: LAPORAN AKHIR
# ==========================================
print("\n" + "="*30)
print("LAPORAN PEMBERSIHAN")
print("="*30)
print(f"Total File Awal      : {len(df_meta)}")
print(f"Jumlah Gambar Valid  : {len(df_meta[df_meta['status_valid']==True])}")
print(f"Jumlah Gambar Dibuang: {len(df_meta[df_meta['status_valid']==False])}")
print(f"Jumlah Duplikat      : {df_meta['alasan_dibuang'].astype(str).str.contains('Duplikat').sum()}")

print("\n--- Distribusi Per Label (Valid) ---")
print(df_meta[df_meta['status_valid']==True]['label'].value_counts())
print("==================================\n")
