import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import cv2
import numpy as 

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

# Adding a new layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # Ek bir tam bağlantılı katman
predictions = Dense(num_classes, activation='softmax')(x)  # Çıkış katmanı

# Creating the model
model = Model(inputs=base_model.input, outputs=predictions)

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Datasets
train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\ruyaa\\OneDrive\\Desktop\\DATASET\\train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'C:\Users\\ruyaa\\OneDrive\\Desktop\\DATASET\\test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Num of classes 
num_classes = len(train_generator.class_indices)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training
history = model.fit(
    train_generator,
    epochs=60,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size
)

base_model.trainable = True

for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Retraining
history_finetune = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size
)

#Starting the webcam
cap = cv2.VideoCapture(1) # 0, bilgisayarın varsayılan kamerasını temsil eder; 1 dahili kamera olmadığı için kullanılır

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    #Normalizing the image 
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0 

    #Perform prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    class_label = list(train_generator.class_indices.keys())[predicted_class]  # Sınıf ismini alın

    #Display the classification result on the frame
    cv2.putText(frame, f"Class: {class_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-Time Object Detection', frame)

    #Exit when the 'r' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('r'):
        break
cap.release()
cv2.destroyAllWindows()