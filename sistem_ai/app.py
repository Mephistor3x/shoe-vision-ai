import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# Impor Model AI secara dinamis untuk mendukung LiteRT / TFLite (tanpa full TensorFlow)
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
    has_tensorflow = True
    print("[+] Berhasil mengimpor TensorFlow (Menggunakan library lengkap).")
except ImportError:
    # Fallback ke ai-edge-litert (atau tflite_runtime) jika full TensorFlow tidak terinstal (misalnya di Hugging Face)
    try:
        import ai_edge_litert.interpreter as tflite
        print("[+] TensorFlow tidak ditemukan. Menggunakan fallback ai-edge-litert (LiteRT).")
    except ImportError:
        import tflite_runtime.interpreter as tflite
        print("[+] TensorFlow tidak ditemukan. Menggunakan fallback tflite-runtime (Ringan).")
    has_tensorflow = False

# Cek apakah sedang berjalan di lingkungan serverless Vercel
IS_VERCEL = os.getenv("VERCEL") == "1"

# =====================================================================
# 1. KONFIGURASI LINGKUNGAN & API KEY GEMINI
# =====================================================================

# Mendapatkan path absolut direktori tempat app.py ini berada
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Memuat file konfigurasi rahasia (.env) untuk mengambil GEMINI_API_KEY secara aman
load_dotenv(os.path.join(BASE_DIR, '.env'))

# Konfigurasi resmi SDK Google Generative AI secara opsional
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        print("[+] API Key Gemini berhasil dikonfigurasi.")
    except Exception as e:
        print(f"[X] Gagal mengonfigurasi Gemini API: {e}")
else:
    print("[!] Peringatan: GEMINI_API_KEY tidak ditemukan di environment. Fallback Gemini akan dinonaktifkan.")

# Inisialisasi aplikasi Flask untuk antarmuka web
app = Flask(__name__)

# =====================================================================
# 2. KONFIGURASI FOLDER UNGGAHAN GAMBAR (UPLOAD)
# =====================================================================

# Menentukan folder penyimpanan sementara untuk foto sepatu yang diunggah
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'jfif'}  # Ekstensi gambar yang didukung
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Membuat folder unggahan otomatis jika belum ada di server (dilewati di Vercel karena read-only)
if not IS_VERCEL:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =====================================================================
# 3. MEMUAT MODEL MACHINE LEARNING LOKAL (TFLITE / KERAS)
# =====================================================================

# Menentukan jalur berkas model hasil pelatihan (MobileNetV3)
MODEL_TFLITE_PATH = os.path.join(BASE_DIR, 'model_merek_sepatu.tflite')
MODEL_KERAS_PATH = os.path.join(BASE_DIR, 'model_merek_sepatu.keras')

is_tflite = False
interpreter = None
input_details = None
output_details = None
model = None

# Membaca model TFLite atau Keras secara aman dengan penanganan kesalahan
try:
    if IS_VERCEL:
        print("[*] Berjalan di Vercel (Serverless). Menonaktifkan model lokal untuk stabilitas penuh...")
        is_tflite = False
    elif os.path.exists(MODEL_TFLITE_PATH):
        print(f"[+] Memuat model TFLite dari: {MODEL_TFLITE_PATH} (Inferensi Cepat)...")
        if has_tensorflow:
            interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE_PATH)
        else:
            interpreter = tflite.Interpreter(model_path=MODEL_TFLITE_PATH)
        interpreter.allocate_tensors()  # Alokasi memori tensor
        input_details = interpreter.get_input_details()    # Detail bentuk input gambar
        output_details = interpreter.get_output_details()  # Detail bentuk output prediksi
        is_tflite = True
    # Membaca model Keras (Sebagai cadangan lokal jika file TFLite tidak ditemukan dan TF terinstal)
    elif has_tensorflow and os.path.exists(MODEL_KERAS_PATH):
        print(f"[+] Memuat model Keras dari: {MODEL_KERAS_PATH} (Fallback)...")
        model = tf.keras.models.load_model(MODEL_KERAS_PATH)
    else:
        print("[!] ERROR: Tidak dapat memuat model pendeteksi sepatu!")
except Exception as model_err:
    print(f"[X] Gagal memuat model klasifikasi lokal: {model_err}")
    print("[!] Sistem akan beroperasi penuh menggunakan Gemini AI sebagai backend utama.")
    is_tflite = False

# Urutan nama kelas target yang dideteksi oleh model lokal kita
class_names = ['adidas', 'lainnya', 'new_balance', 'nike', 'puma']

def allowed_file(filename):
    """
    Fungsi utilitas untuk memeriksa apakah berkas gambar yang diunggah
    memiliki salah satu ekstensi yang diizinkan (PNG, JPG, JPEG, WEBP, JFIF).
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =====================================================================
# 4. FUNGSI PREDIKSI MENGGUNAKAN MODEL LOKAL (TFLITE)
# =====================================================================
def predict_shoe(img_path):
    """
    Melakukan klasifikasi merek sepatu secara lokal menggunakan model MobileNetV3 TFLite.
    
    Proses:
    1. Membuka gambar dan mengubah resolusinya menjadi 224x224 (ukuran wajib untuk MobileNetV3).
    2. Mengonversi gambar menjadi array matriks angka (RGB).
    3. Menambahkan dimensi batch (shape menjadi [1, 224, 224, 3]).
    4. Melakukan normalisasi piksel gambar menggunakan preprocessing bawaan MobileNetV3.
    5. Menjalankan inferensi dan menghitung kelas dengan probabilitas (keyakinan) tertinggi.
    """
    if has_tensorflow:
        # 1. Load & Resize gambar ke 224x224 piksel menggunakan Keras
        img = image.load_img(img_path, target_size=(224, 224))
        # 2. Konversi gambar ke array NumPy
        img_array = image.img_to_array(img)
        # 3. Ubah bentuk array menjadi batch: (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        # 4. Terapkan preprocessing MobileNetV3
        img_array = preprocess_input(img_array)
    else:
        # 1. Load & Resize gambar menggunakan Pillow (tanpa dependensi Keras)
        img = Image.open(img_path).resize((224, 224))
        # Pastikan gambar dalam format RGB (membuang alpha channel jika ada di png/webp)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # 2. Konversi gambar ke array numpy
        img_array = np.array(img, dtype=np.float32)
        # 3. Ubah bentuk array menjadi batch: (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        # MobileNetV3 pada Keras tidak memerlukan scaling piksel tambahan (pass-through)

    if is_tflite:
        # Jalankan inferensi menggunakan interpreter TFLite
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
    else:
        # Jalankan inferensi menggunakan model Keras biasa (hanya jika full TensorFlow terinstal)
        if has_tensorflow and model is not None:
            predictions = model.predict(img_array)
        else:
            raise ValueError("Model Keras tidak didukung tanpa pustaka TensorFlow lengkap!")

    # Menghitung index probabilitas tertinggi
    predicted_index = np.argmax(predictions[0])
    
    # Mendapatkan persentase keyakinan model (skala 0 - 100%)
    confidence = float(predictions[0][predicted_index] * 100)
    
    # Mengembalikan label merek dalam huruf kapital beserta persentase keyakinannya
    return class_names[predicted_index].upper(), confidence

# =====================================================================
# 5. FUNGSI CADANGAN MENGGUNAKAN GEMINI 2.5 FLASH API
# =====================================================================
def predict_shoe_with_gemini(img_path):
    """
    Mengirimkan gambar sepatu ke Gemini 2.5 Flash API jika model lokal ragu-ragu.
    
    Proses:
    1. Membuka gambar menggunakan Pillow.
    2. Menyusun prompt teks yang tegas meminta klasifikasi ke 5 merek sasaran.
    3. Menggunakan fitur Structured Output (response_mime_type: application/json)
       agar Gemini wajib mengembalikan data dalam format JSON yang bersih dan mudah diparsing.
    """
    try:
        # Memeriksa apakah API Key Gemini dikonfigurasi sebelum melakukan panggilan
        if not os.getenv("GEMINI_API_KEY"):
            print("[!] Peringatan: Panggilan Gemini dilewati karena GEMINI_API_KEY tidak dikonfigurasi.")
            return None, None

        # Membuka gambar menggunakan format Pillow PIL agar didukung oleh Gemini SDK
        img = Image.open(img_path)
        
        # Menggunakan model Gemini 2.5 Flash yang sangat cepat dan mendukung input multimodal (gambar+teks)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Instruksi mendetail untuk memandu Gemini
        prompt = """
        Analyze this shoe image. Identify the brand of the shoe.
        You must classify it into one of these exact brand values:
        - ADIDAS
        - NIKE
        - NEW_BALANCE
        - PUMA
        - LAINNYA
        
        Return a JSON object in this exact format:
        {
          "brand": "BRAND_NAME",
          "confidence": 95.0
        }
        Where BRAND_NAME must be one of the brand values listed above (all uppercase, with underscores for NEW_BALANCE), and confidence is a float from 0.0 to 100.0 representing your confidence.
        """
        
        # Mengirimkan prompt dan gambar ke Gemini API
        response = gemini_model.generate_content(
            [prompt, img],
            generation_config={"response_mime_type": "application/json"} # Wajib mengembalikan JSON valid
        )
        
        # Membaca teks hasil respons dan mengubahnya kembali menjadi objek dictionary Python
        data = json.loads(response.text.strip())
        brand = data.get("brand", "LAINNYA").upper()
        confidence = float(data.get("confidence", 95.0))
        
        return brand, confidence
    except Exception as e:
        print(f"[X] Gagal melakukan prediksi dengan Gemini: {e}")
        return None, None

# =====================================================================
# 6. ROUTING ROUTE (PENGATURAN ALUR WEB & API FLASK)
# =====================================================================

@app.route('/')
def index():
    """
    Halaman Utama: Merender template index.html yang berisi layout drag & drop
    serta antarmuka visual unggah gambar sepatu.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint POST API /predict:
    Menerima berkas foto dari client, menyimpannya, lalu mengeksekusi logika Hybrid AI.
    
    Alur Hybrid AI (Scheme 1):
    1. Lakukan prediksi lokal menggunakan TFLite.
    2. Periksa apakah tingkat keyakinan (confidence) di bawah 60.00% ATAU terdeteksi sebagai "LAINNYA".
       - Jika Ya: Panggil Gemini API untuk menganalisis gambar secara mendalam. Ganti hasil prediksi lokal.
       - Jika Tidak: Gunakan hasil prediksi lokal (hemat kuota API dan sangat cepat).
    3. Kembalikan data dalam format JSON ke client.
    """
    # Validasi apakah berkas dikirim dalam request POST
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diupload'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'})
        
    print(f"[?] Menerima file: '{file.filename}'")
    if file and allowed_file(file.filename):
        # Amankan nama file dari karakter aneh
        filename = secure_filename(file.filename)
        
        # Di Vercel, kita simpan di folder /tmp karena filesystem lainnya read-only
        if IS_VERCEL:
            filepath = os.path.join('/tmp', filename)
        else:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
        print(f"[!] Menghandle file: {filename}")
        print(f"[!] Path simpan: {filepath}")
        
        file.save(filepath)
        
        # Lakukan prediksi
        try:
            # Langkah 1: Prediksi menggunakan Model TFLite lokal (jika berhasil dimuat)
            label = None
            confidence = None
            used_gemini = False
            
            if is_tflite or (has_tensorflow and model is not None):
                try:
                    label, confidence = predict_shoe(filepath)
                    print(f"[+] Prediksi Lokal: {label} ({confidence:.2f}%)")
                except Exception as local_err:
                    print(f"[X] Gagal melakukan prediksi lokal: {local_err}. Mencoba fallback ke Gemini...")
            else:
                print("[!] Model lokal tidak tersedia. Mencoba menggunakan Gemini AI secara langsung...")
            
            # Langkah 2: Pemicu Fallback Gemini (Keyakinan rendah, terdeteksi "LAINNYA", atau model lokal tidak tersedia)
            if label is None or confidence < 60.00 or label == "LAINNYA":
                print(f"[!] Memicu Gemini (Alasan: model lokal tidak tersedia/keyakinan rendah/LAINNYA)...")
                gemini_label, gemini_confidence = predict_shoe_with_gemini(filepath)
                
                # Jika pemanggilan Gemini sukses, gunakan hasilnya
                if gemini_label is not None:
                    label = gemini_label
                    confidence = gemini_confidence
                    used_gemini = True
                    print(f"[GEMINI] Hasil Gemini: {label} ({confidence:.2f}%)")
                else:
                    if label is None:
                        raise ValueError("Gagal mendeteksi gambar: Model lokal tidak tersedia dan panggilan Gemini AI gagal.")
                    else:
                        print("[!] Panggilan Gemini gagal. Tetap menggunakan hasil prediksi lokal.")
            
            # Langkah 3: Kirim respons JSON akhir ke frontend
            return jsonify({
                'success': True,
                'label': label,
                'confidence': f"{confidence:.2f}",
                'image_url': f"/static/uploads/{filename}" if not IS_VERCEL else "",
                'used_gemini': used_gemini
            })
        except Exception as e:
            print(f"[X] Error Prediksi: {str(e)}")
            return jsonify({'error': str(e)})
            
    return jsonify({'error': f'Format file tidak didukung: {file.filename}'})

# =====================================================================
# 7. MENJALANKAN FLASK SERVER
# =====================================================================
if __name__ == '__main__':
    # Membaca port dinamis dari sistem (port 7860 wajib di Hugging Face, default 5000 untuk lokal)
    port = int(os.environ.get("PORT", 5000))
    # Menjalankan aplikasi pada host 0.0.0.0 agar bisa diakses secara publik/cloud hosting
    app.run(debug=True, host='0.0.0.0', port=port)
