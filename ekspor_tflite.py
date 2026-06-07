import os
import tensorflow as tf

def convert_model(keras_path, tflite_path):
    if not os.path.exists(keras_path):
        print(f"[!] File model Keras '{keras_path}' tidak ditemukan. Melewati...")
        return False
        
    print(f"[+] Memuat model Keras dari: {keras_path}...")
    try:
        model = tf.keras.models.load_model(keras_path)
    except Exception as e:
        print(f"[X] Gagal memuat model Keras: {e}")
        return False

    print("[+] Mengonversi model ke format TFLite...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # Aktifkan optimasi default untuk kompresi ukuran model
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
    except Exception as e:
        print(f"[X] Gagal melakukan konversi TFLite: {e}")
        return False
    
    print(f"[+] Menyimpan model TFLite ke: {tflite_path}...")
    try:
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"[+] Konversi berhasil! Model TFLite disimpan di: {tflite_path}")
        keras_size = os.path.getsize(keras_path) / (1024 * 1024)
        tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"    - Ukuran Keras: {keras_size:.2f} MB")
        print(f"    - Ukuran TFLite: {tflite_size:.2f} MB")
        print(f"    - Penghematan: {((keras_size - tflite_size) / keras_size) * 100:.1f}%")
        return True
    except Exception as e:
        print(f"[X] Gagal menyimpan model TFLite: {e}")
        return False

if __name__ == "__main__":
    print("=== SKRIP KONVERSI MODEL KERAS KE TFLITE ===")
    
    # Konversi model di dalam sistem_ai
    app_model_keras = os.path.join("sistem_ai", "model_merek_sepatu.keras")
    app_model_tflite = os.path.join("sistem_ai", "model_merek_sepatu.tflite")
    convert_model(app_model_keras, app_model_tflite)
    
    print("\n" + "-"*40 + "\n")
    
    # Konversi model di root jika ada
    root_model_keras = "model_resnet_sepatu.keras"
    root_model_tflite = "model_resnet_sepatu.tflite"
    convert_model(root_model_keras, root_model_tflite)
