import os
import shutil
import splitfolders

DIR_SUMBER = 'dataset_clean'
DIR_TUJUAN = 'dataset_merek'
DIR_FINAL = 'dataset_merek_final'

# Kata kunci merek yang ingin kita klasifikasikan
merek_target = ['adidas', 'nike', 'new_balance', 'puma']

print(f"[*] Mengelompokkan ulang dataset dari {DIR_SUMBER} ke {DIR_TUJUAN}...\n")

if not os.path.exists(DIR_TUJUAN):
    os.makedirs(DIR_TUJUAN)

for merek in merek_target:
    os.makedirs(os.path.join(DIR_TUJUAN, merek), exist_ok=True)
os.makedirs(os.path.join(DIR_TUJUAN, 'lainnya'), exist_ok=True)

jumlah_per_merek = {m: 0 for m in merek_target + ['lainnya']}

# Pindai semua folder label yang panjang
if os.path.exists(DIR_SUMBER):
    for nama_folder in os.listdir(DIR_SUMBER):
        path_folder = os.path.join(DIR_SUMBER, nama_folder)
        
        if os.path.isdir(path_folder):
            # Tentukan merek berdasarkan nama folder
            merek_ditemukan = 'lainnya'
            nama_folder_lower = nama_folder.lower()
            
            for merek in merek_target:
                if merek in nama_folder_lower:
                    merek_ditemukan = merek
                    break
                    
            # Copy semua gambar di folder ini ke folder merek yang sesuai
            for nama_file in os.listdir(path_folder):
                file_sumber = os.path.join(path_folder, nama_file)
                
                # Buat nama unik agar tidak tertimpa
                file_tujuan = os.path.join(DIR_TUJUAN, merek_ditemukan, f"{nama_folder[:10]}_{nama_file}")
                
                if os.path.isfile(file_sumber):
                    shutil.copy2(file_sumber, file_tujuan)
                    jumlah_per_merek[merek_ditemukan] += 1
else:
    print(f"[!] Folder sumber '{DIR_SUMBER}' tidak ditemukan. Jalankan 'bersihkan_dataset.py' terlebih dahulu.")

print("\n[+] Pengelompokan Selesai!")
print("--- JUMLAH GAMBAR PER MEREK ---")
for merek, jumlah in jumlah_per_merek.items():
    print(f"- {merek.upper():<15}: {jumlah} gambar")
print("===============================\n")

# Pembagian split folder untuk latih_resnet
print("[+] Membagi dataset merek menjadi Train, Val, dan Test...")
if os.path.exists(DIR_TUJUAN) and len(os.listdir(DIR_TUJUAN)) > 0:
    splitfolders.ratio(DIR_TUJUAN, output=DIR_FINAL, seed=42, ratio=(.7, .2, .1))
    print(f"[+] Pembagian selesai! Folder '{DIR_FINAL}' siap digunakan untuk training.")
else:
    print("[!] Tidak ada gambar valid di folder merek untuk di-split.")
