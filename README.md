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

Sistem klasifikasi merek sepatu berbasis **Hybrid AI** yang memadukan model lokal ringan **TensorFlow Lite (MobileNetV3-Large)** dengan kekuatan model cloud **Gemini 2.5 Flash API** sebagai fallback otomatis.

Sistem ini dirancang untuk mencapai akurasi maksimal dengan efisiensi biaya/kuota API yang tinggi dengan hanya memanggil Gemini ketika model lokal ragu-ragu.

---

## 🚀 Fitur Utama
1. **Model Klasifikasi Lokal Cepat**: Menggunakan model MobileNetV3-Large yang dikompresi ke format TFLite (hanya **3.57 MB**) untuk inferensi instan di server.
2. **Fallback Dinamis (Gemini 2.5 Flash)**: Mengalihkan analisis secara realtime ke Gemini API jika keyakinan model lokal di bawah **60%** atau menebak kelas `"LAINNYA"`.
3. **UI Modern & Premium**: Tampilan futuristik berbasis *glassmorphism*, dilengkapi lencana **⚡ Gemini AI Verified** beranimasi nyala (*glow pulse*) serta garis sensor laser scan.
4. **Keamanan Kunci API**: Konfigurasi aman menggunakan variabel lingkungan (`.env`) agar API key tidak terekspos ke repositori Git.
5. **Siap Cloud Deployment**: Dilengkapi dengan `Dockerfile` standar produksi untuk deployment instan ke Hugging Face Spaces.

---

## 🛠️ Cara Menjalankan secara Lokal

### 1. Prasyarat
Pastikan Python 3.9+ telah terpasang di komputer Anda.

### 2. Pemasangan Dependensi
Buka terminal di direktori proyek dan jalankan perintah:
```bash
pip install -r sistem_ai/requirements.txt
```

### 3. Konfigurasi Kunci API Gemini
Buat file bernama `.env` di dalam folder `sistem_ai` (atau salin dari template jika ada) dan isi dengan kunci API Gemini Anda:
```env
GEMINI_API_KEY=KUNCI_API_GEMINI_ANDA
```
*Catatan: File `.env` ini secara otomatis diabaikan oleh Git agar tidak terunggah ke internet.*

### 4. Menjalankan Aplikasi
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
4. **Penting**: Masuk ke menu **Settings** di Hugging Face Space Anda, lalu tambahkan **Variable / Secret** baru bernama `GEMINI_API_KEY` dan isi dengan kunci API Gemini Anda. Ini menggantikan file `.env` lokal Anda dengan aman di cloud.
