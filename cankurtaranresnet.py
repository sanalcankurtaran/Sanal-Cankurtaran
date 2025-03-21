import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
from ultralytics import YOLO 

# YOLO Model
yolo_model = YOLO("yolov8n.pt")  # YOLOv8 Nano 

IMAGE_SIZE = [224, 224]
trainingFolder = "C:\\Users\\ruyaa\\OneDrive\\Desktop\\DATASET\\train"  #Training dataset
testFolder = "C:\\Users\\ruyaa\\OneDrive\\Desktop\\DATASET\\test"    #Test dataset

# Pretrained ResNet50 model
myResnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights="imagenet", include_top=False)

# Freeze the layers of the base model
for layer in myResnet.layers:
    layer.trainable = False

Classes = os.listdir(trainingFolder) 
numOfClasses = len(Classes)  

# Add layers to the model
global_avg_pooling_layer = tf.keras.layers.GlobalAveragePooling2D()(myResnet.output)
PlusFlattenLayer = Flatten()(global_avg_pooling_layer)
predictionLayer = Dense(numOfClasses, activation='softmax')(PlusFlattenLayer)
model = Model(inputs=myResnet.input, outputs=predictionLayer)

#Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy']
)

#Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

#Load the datasets
training_set = train_datagen.flow_from_directory(
    trainingFolder,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    testFolder,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

#Training
EPOCHS = 20  
model.fit(
    training_set,
    validation_data=test_set,
    epochs=EPOCHS,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

#Save the model 
model.save("model.h5")  # Modeli HDF5 formatında kaydeder

#TensorFlow Lite

model = tf.keras.models.load_model("model.h5")

#Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model başarıyla TFLite formatına dönüştürüldü")

#Start the webcam
cap = cv2.VideoCapture(1)  # 0, bilgisayarın varsayılan kamerasını temsil eder; 1 dahili kamera olmadığı için kullanılır

#Create a mapping of class indices to class names
class_map = {v: k for k, v in training_set.class_indices.items()}

while True:
    ret, frame = cap.read()
    if not ret:
        break

 # Object Detection with YOLO
    results = yolo_model.predict(frame, save=False)  
    for result in results[0].boxes: 
        x1, y1, x2, y2 = map(int, result.xyxy[0])  
        cropped_object = frame[y1:y2, x1:x2] 

 #Invalid bounding box check
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        continue  # Eğer bounding box geçersizse, bu bounding box'ı atla
       
        cropped_object = frame[y1:y2, x1:x2]  
        
        #Classification with ResNet50
        if cropped_object.size > 0:  # Çerçeve geçerli mi kontrol et
            resized_cropped_object = cv2.resize(cropped_object, (224, 224))  
            normalized_object = resized_cropped_object / 255.0  
            input_object = np.expand_dims(normalized_object, axis=0)  

            predictions = model.predict(input_object)  
            class_idx = np.argmax(predictions)  
            class_label = class_map[class_idx]  

            #Visualize Detection and Classification Results
            cv2.putText(frame, f"{class_label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("YOLO + ResNet50 Detection", frame)

    resized_frame = cv2.resize(frame, (224, 224))  
    normalized_frame = resized_frame / 255.0  
    input_frame = np.expand_dims(normalized_frame, axis=0)  

    #Perform prediction
    predictions = model.predict(input_frame)
    class_idx = np.argmax(predictions)  # En yüksek olasılıklı sınıfı alır
    class_label = class_map[class_idx]  # Sınıf ismini alır

    #Display the classification result on the frame
    cv2.putText(frame, f"Class: {class_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Classification", frame)

    #Exit when the 'r' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('r'):
        break

cap.release()
cv2.destroyAllWindows()
