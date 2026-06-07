---
title: Shoe Vision AI
emoji: 👟
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Shoe Vision AI 👟

Sistem klasifikasi merek sepatu berbasis **Deep Learning** lokal menggunakan model **TensorFlow Lite (MobileNetV3-Large)** yang berjalan sepenuhnya di server tanpa dependensi cloud API eksternal.

Sistem ini dirancang untuk mendeteksi merek-merek sepatu ternama seperti Adidas, Nike, New Balance, Puma, atau mengkategorikannya sebagai Lainnya secara instan, aman, dan tanpa biaya operasional API.

---

## 🚀 Fitur Utama
1. **Model Klasifikasi Lokal Cepat**: Menggunakan model MobileNetV3-Large yang dikompresi ke format TFLite (hanya **3.57 MB**) untuk inferensi instan.
2. **UI Modern & Premium**: Tampilan futuristik berbasis *glassmorphism*, dilengkapi visualisasi hasil klasifikasi interaktif, persentase keyakinan, serta garis sensor laser scan.
3. **Penyimpanan Gambar Temporer Vercel**: Dukungan penanganan upload gambar temporer di folder `/tmp` untuk kompatibilitas penuh dengan platform serverless seperti Vercel.
4. **Siap Cloud Deployment**: Dilengkapi dengan `Dockerfile` standar produksi untuk deployment instan ke Hugging Face Spaces.

---

## 🛠️ Cara Menjalankan secara Lokal

### 1. Prasyarat
Pastikan Python 3.9+ telah terpasang di komputer Anda.

### 2. Pemasangan Dependensi
Buka terminal di direktori proyek dan jalankan perintah:
```bash
pip install -r sistem_ai/requirements.txt
```

### 3. Menjalankan Aplikasi
Jalankan Flask server dengan perintah:
```bash
python sistem_ai/app.py
```
Buka browser Anda dan akses halaman web di: **`http://127.0.0.1:5000`**

---

## 🐳 Panduan Deployment ke Hugging Face Spaces

1. Buat Space baru di [Hugging Face](https://huggingface.co/new-space).
2. Tulis nama Space Anda, lalu pilih **Docker** sebagai SDK (Hugging Face akan otomatis membaca berkas `Dockerfile` kita).
3. Hubungkan repositori GitHub Anda ke Space tersebut, atau lakukan push langsung ke remote git Hugging Face.
