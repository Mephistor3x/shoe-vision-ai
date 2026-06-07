import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# ==========================================
# KONFIGURASI PATH
# ==========================================
TRAIN_DIR = os.path.join('dataset_merek_final', 'train')

# Model candidate paths (TFLite preferred, Keras fallback)
MODEL_CANDIDATES = [
    # TFLite Paths
    ('model_merek_sepatu.tflite', 'tflite'),
    (os.path.join('sistem_ai', 'model_merek_sepatu.tflite'), 'tflite'),
    ('model_resnet_sepatu.tflite', 'tflite'),
    # Keras Paths
    ('model_merek_sepatu.keras', 'keras'),
    (os.path.join('sistem_ai', 'model_merek_sepatu.keras'), 'keras'),
    ('model_resnet_sepatu.keras', 'keras')
]

# Mendapatkan daftar label kelas (fallback jika folder train tidak ditemukan)
if os.path.exists(TRAIN_DIR):
    class_names = sorted(os.listdir(TRAIN_DIR))
else:
    class_names = ['adidas', 'lainnya', 'new_balance', 'nike', 'puma']

# ==========================================
# FUNGSI PREDIKSI
# ==========================================
def prediksi_gambar(img_path):
    """
    Memuat model yang ada (TFLite atau Keras), lalu memprediksi merek sepatu.

    Parameter:
    - img_path: path gambar sepatu yang ingin ditebak.
    """
    # Cari model yang tersedia
    model_path = None
    model_type = None
    
    for path, mtype in MODEL_CANDIDATES:
        if os.path.exists(path):
            model_path = path
            model_type = mtype
            break
            
    if model_path is None:
        print("[!] Error: Tidak ada model (.tflite atau .keras) yang ditemukan.")
        print("    Jalankan 'latih_resnet.py' atau 'ekspor_tflite.py' terlebih dahulu.")
        return

    # Load dan preprocess gambar (Resize ke 224x224)
    try:
        img = image.load_img(img_path, target_size=(224, 224))
    except Exception as e:
        print(f"[!] Error membuka gambar {img_path}: {e}")
        return

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    if model_type == 'tflite':
        print(f"[+] Memuat model TFLite dari: {model_path} (Inferensi Cepat)...")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Jalankan prediksi dengan TFLite
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
    else:
        print(f"[+] Memuat model Keras dari: {model_path} (ini mungkin butuh beberapa detik)...")
        model = tf.keras.models.load_model(model_path)
        
        # Jalankan prediksi dengan Keras
        predictions = model.predict(img_array)

    # Ambil probabilitas tertinggi
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index] * 100
    predicted_label = class_names[predicted_index]

    print("\n" + "="*40)
    print("HASIL PREDIKSI AI:")
    print("="*40)
    print(f"File Gambar : {os.path.basename(img_path)}")
    print(f"Hasil Tebak : {predicted_label.upper()}")
    print(f"Keyakinan   : {confidence:.2f}%")
    print(f"Menggunakan : Model {model_type.upper()} ({os.path.basename(model_path)})")
    print("="*40)

# ==========================================
# JALANKAN PREDIKSI
# ==========================================
if __name__ == "__main__":
    print("\n--- PROGRAM PREDIKSI SEPATU ---")
    gambar_test = input("Masukkan path/nama file gambar sepatu yang ingin ditebak: ")
    
    if os.path.exists(gambar_test):
        prediksi_gambar(gambar_test)
    else:
        print("[!] File tidak ditemukan. Cek kembali path atau nama filenya.")
