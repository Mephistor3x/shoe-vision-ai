import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ==========================================
# KONFIGURASI TRAINING (TURBO VERSION)
# ==========================================
DATASET_DIR = 'dataset_merek_final'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10    # Tahap 1: Melatih kepala model
FINE_TUNE_EPOCHS = 20   # Tahap 2: Melatih seluruh model (Fine-Tuning)
MODEL_SAVE_PATH = 'model_merek_sepatu.keras'

# ==========================================
# LANGKAH 1: AUGMENTASI LEBIH KUAT
# ==========================================
print("[+] Menyiapkan Data Generator (Augmentasi Turbo)...")

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input,
    rotation_range=30,           # Lebih berani memutar gambar
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2], # Simulasi pencahayaan berbeda
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

NUM_CLASSES = train_generator.num_classes

# ==========================================
# LANGKAH 2: MEMBANGUN STRUKTUR MODEL
# ==========================================
print("\n[+] Membangun Arsitektur ResNet50V2...")

base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Tahap 1: Bekukan base model
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x) # Tambahan: Stabilisasi distribusi data
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ==========================================
# LANGKAH 3: TAHAP 1 - TRAINING KEPALA (WARMUP)
# ==========================================
print(f"\n[🚀 TAHAP 1] Memulai Warmup Training (Kepala Model)...")

callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
]

history_warmup = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks
)

# ==========================================
# LANGKAH 4: TAHAP 2 - FINE TUNING (UNFREEZE)
# ==========================================
print(f"\n[🔥 TAHAP 2] Memulai Fine-Tuning (Membuka Layer ResNet)...")

# Kita unfreeze (buka kunci) base model
base_model.trainable = True

# PENTING: Gunakan Learning Rate SANGAT KECIL saat fine-tuning
# Agar tidak merusak bobot yang sudah pintar dari ImageNet
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINE_TUNE_EPOCHS,
    initial_epoch=history_warmup.epoch[-1], # Lanjutkan dari epoch terakhir
    callbacks=callbacks
)

print("\n[+] Training Selesai! Model versi Turbo telah disimpan.")

# ==========================================
# LANGKAH 5: VISUALISASI TRAINING HISTORY
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
    plt.title('Akurasi Training & Validasi', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='#f87171', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='#f59e0b', linewidth=2)
    plt.axvline(x=warmup_epochs, color='#ef4444', linestyle='--', linewidth=1.5, label='Mulai Fine-Tuning')
    plt.title('Loss Training & Validasi', fontsize=12, fontweight='bold')
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
