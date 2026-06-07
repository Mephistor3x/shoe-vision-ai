import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from collections import Counter

# ==========================================
# KONFIGURASI TRAINING (CPU-OPTIMIZED VERSION)
# ==========================================
DATASET_DIR = 'dataset_merek_final'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10     # Tahap 1: Melatih kepala model (Warmup)
FINE_TUNE_EPOCHS = 20    # Tahap 2: Melatih seluruh model (Fine-Tuning)
MODEL_SAVE_PATH = 'model_merek_sepatu.keras'

# ==========================================
# LANGKAH 1: DATA GENERATOR & AUGMENTASI
# ==========================================
print("[+] Menyiapkan Data Generator (MobileNetV3 Preprocessing)...")

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input
)

train_path = os.path.join(DATASET_DIR, 'train')
val_path = os.path.join(DATASET_DIR, 'val')

if not os.path.exists(train_path) or not os.path.exists(val_path):
    print(f"[!] Error: Folder dataset '{train_path}' atau '{val_path}' tidak ditemukan.")
    print("    Pastikan Anda sudah menjalankan 'kelompokkan_merek.py' terlebih dahulu.")
    exit(1)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

NUM_CLASSES = train_generator.num_classes

# ==========================================
# LANGKAH 2: MENGHITUNG CLASS WEIGHTS (IMBALANCE)
# ==========================================
class_counts = Counter(train_generator.classes)
total_samples = sum(class_counts.values())
class_weights = {}

for idx in range(NUM_CLASSES):
    count = class_counts.get(idx, 0)
    if count > 0:
        class_weights[idx] = total_samples / (NUM_CLASSES * count)
    else:
        class_weights[idx] = 1.0  # Bobot default untuk kelas kosong

print("\n" + "="*40)
print("DISTRIBUSI DATA & BOBOT KELAS:")
print("="*40)
for name, idx in sorted(train_generator.class_indices.items(), key=lambda x: x[1]):
    count = class_counts.get(idx, 0)
    weight = class_weights[idx]
    status_label = "" if count > 0 else " (KOSONG - Melewati)"
    print(f"- {name.upper():<15}: {count:<4} gambar (Bobot: {weight:.2f}){status_label}")
print("="*40 + "\n")

# ==========================================
# LANGKAH 3: MEMBANGUN ARSITEKTUR MODEL (MOBILENETV3)
# ==========================================
print("[+] Membangun Arsitektur MobileNetV3-Large...")

# Gunakan MobileNetV3Large yang ringan dan hemat CPU
base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Tahap 1: Bekukan base model
from tensorflow.keras.regularizers import l2
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

# Dense Layer 1 + Regularization
x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Dense Layer 2 + Regularization
x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

# ==========================================
# LANGKAH 4: TAHAP 1 - TRAINING KEPALA (WARMUP)
# ==========================================
print(f"\n[TAHAP 1] Memulai Warmup Training (Kepala Model)...")

callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
]

history_warmup = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ==========================================
# LANGKAH 5: TAHAP 2 - FINE TUNING (UNFREEZE)
# ==========================================
print(f"\n[TAHAP 2] Memulai Fine-Tuning (Membuka Layer MobileNet)...")

# Kita unfreeze base model secara bertahap (Gradual Unfreezing)
base_model.trainable = True
# Bekukan semua layer kecuali 40 layer terakhir
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Gunakan Learning Rate kecil agar tidak merusak bobot pintar ImageNet
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINE_TUNE_EPOCHS,
    initial_epoch=history_warmup.epoch[-1] + 1, # Lanjutkan dari epoch terakhir
    class_weight=class_weights,
    callbacks=callbacks
)

print("\n[+] Training Selesai! Model MobileNetV3-Large telah disimpan.")

# ==========================================
# LANGKAH 6: VISUALISASI TRAINING HISTORY
# ==========================================
print("[+] Membuat grafik pelatihan...")
try:
    acc = history_warmup.history['accuracy'] + history_finetune.history['accuracy']
    val_acc = history_warmup.history['val_accuracy'] + history_finetune.history['val_accuracy']
    loss = history_warmup.history['loss'] + history_finetune.history['loss']
    val_loss = history_warmup.history['val_loss'] + history_finetune.history['val_loss']
    
    epochs_range = range(1, len(acc) + 1)
    warmup_epochs = len(history_warmup.history['accuracy'])

    plt.figure(figsize=(14, 5))

    # Plot Akurasi
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='#38bdf8', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='#818cf8', linewidth=2)
    plt.axvline(x=warmup_epochs, color='#ef4444', linestyle='--', linewidth=1.5, label='Mulai Fine-Tuning')
    plt.title('Akurasi Training & Validasi (MobileNet)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='#f87171', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='#f59e0b', linewidth=2)
    plt.axvline(x=warmup_epochs, color='#ef4444', linestyle='--', linewidth=1.5, label='Mulai Fine-Tuning')
    plt.title('Loss Training & Validasi (MobileNet)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    grafik_path = 'grafik_training.png'
    plt.savefig(grafik_path, dpi=300)
    print(f"[+] Grafik pelatihan berhasil disimpan di: {grafik_path}")
except Exception as e:
    print(f"[X] Gagal membuat grafik pelatihan: {e}")
