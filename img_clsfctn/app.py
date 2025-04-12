import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

# ✅ Update dataset path
dataset_path = r"D:\DL\img_clsfctn\image-forgery-detection"  # Adjusted to match actual structure
train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "val")

# Step 2: Check if dataset exists
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    print("❌ ERROR: Train or Validation directory not found!")
    print("📂 Checking extracted dataset structure...\n")

    for root, dirs, files in os.walk(dataset_path):
        print(f"📁 {root}")
        for d in dirs:
            print(f"  📂 {d}")
        for f in files:
            print(f"  📄 {f}")

    raise FileNotFoundError("Dataset folders not found. Please verify the dataset structure.")

print("✅ Dataset found. Proceeding with training...")

# Step 3: Image Preprocessing & Augmentation
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode='binary', subset='training')

val_generator = datagen.flow_from_directory(
    val_dir, target_size=(150, 150), batch_size=32, class_mode='binary', subset='validation')

# Step 4: Define Custom CNN Model
cnn_model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n🚀 Training Custom CNN Model...")
cnn_model.fit(train_generator, validation_data=val_generator, epochs=10)
cnn_model.save("cnn_forgery_model.h5")
print("✅ Custom CNN model saved as 'cnn_forgery_model.h5'\n")

# Step 5: Transfer Learning with VGG16
print("\n🚀 Training VGG16 Model...")
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
for layer in vgg_base.layers:
    layer.trainable = False  # Freeze VGG16 layers

vgg_model = keras.Sequential([
    vgg_base,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

vgg_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
vgg_model.fit(train_generator, validation_data=val_generator, epochs=10)
vgg_model.save("vgg_forgery_model.h5")
print("✅ VGG16 model saved as 'vgg_forgery_model.h5'")

print("\n🎉 All models trained and saved successfully!")
