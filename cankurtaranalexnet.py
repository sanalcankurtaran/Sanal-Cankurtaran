import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

# Dataset
train_dir = "C:\\Users\\ruyaa\\OneDrive\\Desktop\\DATASET\\train"  # Train dataset
val_dir = "C:\\Users\\ruyaa\\OneDrive\\Desktop\\DATASET\\test"      # Test dataset

# Alexnet architecture
def AlexNet(input_shape=(224, 224, 3), num_classes=3):
    model = models.Sequential()
    # 1. Convolutional Layer
    model.add(layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 2. Convolutional Layer
    model.add(layers.Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3., 4., 5. Convolutional Layers
    model.add(layers.Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Çıkış Katmanı
    return model

# Num of classes
num_classes = len(os.listdir(train_dir))

# Building the model
model = AlexNet(input_shape=(224, 224, 3), num_classes=num_classes)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load the datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Training
EPOCHS = 15
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# Save the model
model.save("alexnet_model.h5")
print("Model başarıyla kaydedildi!")

# Start the webcam
cap = cv2.VideoCapture(1)  # 0, bilgisayarın varsayılan kamerasını temsil eder; 1 dahili kamera olmadığı için kullanılır

# Class names
class_names = list(train_generator.class_indices.keys())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Normalizing the image
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Perform prediction
    predictions = model.predict(input_frame)
    class_idx = np.argmax(predictions)
    class_label = class_names[class_idx]

    # Display the classification result on the frame
    cv2.putText(frame, f"Class: {class_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Object Detection", frame)

    # Exit when the 'r' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('r'):
        break

cap.release()
cv2.destroyAllWindows()
