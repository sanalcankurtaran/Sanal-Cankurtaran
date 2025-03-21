import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

#Dataset
trainingFolder = "C:\\Users\\ruyaa\\OneDrive\\Desktop\\DATASET\\train"  #train dataset
testFolder = "C:\\Users\\ruyaa\\OneDrive\\Desktop\\DATASET\\test"    #test dataset

#Define the SqueezeNet architecture
def SqueezeNet(input_shape=(224, 224, 3), num_classes=3):
    input_layer = layers.Input(shape=input_shape)

    # Conv1
    x = layers.Conv2D(96, (7, 7), strides=(2, 2), activation='relu')(input_layer)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Fire Modules
    def fire_module(x, squeeze_filters, expand_filters):
        squeeze = layers.Conv2D(squeeze_filters, (1, 1), activation='relu')(x)
        expand1x1 = layers.Conv2D(expand_filters, (1, 1), activation='relu')(squeeze)
        expand3x3 = layers.Conv2D(expand_filters, (3, 3), padding='same', activation='relu')(squeeze)
        return layers.Concatenate()([expand1x1, expand3x3])

    x = fire_module(x, 16, 64)
    x = fire_module(x, 16, 64)
    x = fire_module(x, 32, 128)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = fire_module(x, 32, 128)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 64, 256)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = fire_module(x, 64, 256)

    # Final Conv Layer
    x = layers.Conv2D(num_classes, (1, 1), activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    output_layer = layers.Activation('softmax')(x)

    return Model(inputs=input_layer, outputs=output_layer)

#Building the model 
IMAGE_SIZE = [224, 224]
NUM_CLASSES = len(os.listdir(trainingFolder))  # Sınıf sayısını klasörlerden alır
model = SqueezeNet(input_shape=IMAGE_SIZE + [3], num_classes=NUM_CLASSES)

#Compile the model 
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

print(model.summary())

#Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    trainingFolder,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    testFolder,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

class_names = list(training_set.class_indices.keys())
print("Sınıf İsimleri:", class_names)

#Train
EPOCHS = 20  
model.fit(
    training_set,
    validation_data=test_set,
    epochs=EPOCHS,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

#Save the model
model.save("squeezenet_model.h5")
print("Model başarıyla kaydedildi!")

#Start the webcam
cap = cv2.VideoCapture(1)  # 0, varsayılan kamerayı temsil eder;  1 dahili kamera olmadığı için kullanılır

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)

    #Perform prediction
    predictions = model.predict(input_frame)
    class_idx = np.argmax(predictions)
    class_label = class_names[class_idx]

    #Display the classification result on the frame
    cv2.putText(frame, f"Class: {class_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Object Detection", frame)

    # Exit when the 'r' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('r'):
        break

cap.release()
cv2.destroyAllWindows()
