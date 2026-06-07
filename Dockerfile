# Menggunakan image base Python 3.11 slim yang ringan untuk meminimalkan ukuran image Docker
FROM python:3.11-slim

# Membuat user non-root dengan UID 1000 (aturan wajib keamanan di Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Mengatur direktori kerja utama di dalam container
WORKDIR /app

# Menyalin berkas requirements.txt secara terpisah agar proses download library di-cache oleh Docker
COPY sistem_ai/requirements.txt /app/requirements.txt

# Melakukan upgrade pip dan menginstal seluruh pustaka Python yang dibutuhkan
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh direktori sistem_ai dan mengubah kepemilikannya ke user non-root (user:user)
COPY --chown=user:user sistem_ai /app/sistem_ai

# Mengubah user aktif container ke non-root untuk mematuhi aturan Hugging Face Spaces
USER user

# Menyetel port default aplikasi ke 7860 (port standar Hugging Face)
ENV PORT=7860

# Membuka port 7860 agar container bisa diakses oleh jaringan luar
EXPOSE 7860

# Perintah utama untuk menjalankan server Flask saat container dimulai
CMD ["python", "sistem_ai/app.py"]
