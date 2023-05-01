# importing the libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


import os
import shutil

# Set the path of the directory containing the images
dir_path = 'C:\\Users\\malli\\Downloads\\Dataset_Real_vs_Fake_image_BinaryClassifier\\images_from_video_big'

# Set the paths of the two new directories for edited and unedited images
edited_dir_path = os.path.join(dir_path, 'C:\\Users\\malli\\Downloads\\Dataset_Real_vs_Fake_image_BinaryClassifier\\images_from_video_big\\edited')
unedited_dir_path = os.path.join(dir_path, 'C:\\Users\\malli\\Downloads\\Dataset_Real_vs_Fake_image_BinaryClassifier\\images_from_video_big\\unedited')

# Create the new directories if they don't already exist
if not os.path.exists(edited_dir_path):
    os.makedirs(edited_dir_path)
if not os.path.exists(unedited_dir_path):
    os.makedirs(unedited_dir_path)

# Iterate over all files in the original directory
for filename in os.listdir(dir_path):
    # Check if the file is a JPG image
    if filename.endswith('.jpg'):
        # Check the last character of the file name to determine if it is edited or unedited
        if filename[-5] == '0':
            # Move the file to the unedited directory
            shutil.move(os.path.join(dir_path, filename), os.path.join(unedited_dir_path, filename))
        elif filename[-5] == '1':
            # Move the file to the edited directory
            shutil.move(os.path.join(dir_path, filename), os.path.join(edited_dir_path, filename))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set the paths of the edited and unedited image directories
edited_dir = 'C:\\Users\\malli\\Downloads\\Dataset_Real_vs_Fake_image_BinaryClassifier\\images_from_video_big\\edited'
unedited_dir = 'C:\\Users\\malli\\Downloads\\Dataset_Real_vs_Fake_image_BinaryClassifier\\images_from_video_big\\unedited'

# Set the image dimensions and batch size
img_height = 128
img_width = 128
batch_size = 32

# Create the training and validation data generators
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dir_path,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',

    seed=123
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dir_path,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=123
)

# Combine the training and validation data for a larger training set
train_ds = train_ds.concatenate(val_ds)

# Create the test data generator
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dir_path,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Create the CNN model
model = keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_ds, epochs=15)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_ds)
print('Test accuracy:', accuracy)

# output-
# "C:\Users\malli\Desktop\py4e\PYTHON\sem pycharm\venv\Scripts\python.exe" "C:\Users\malli\Desktop\py4e\PYTHON\sem pycharm\Alcovex\Alcovex_task-2.py"
# Found 82621 files belonging to 2 classes.
# Using 66097 files for training.
# 2023-05-01 09:57:31.692890: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Found 82621 files belonging to 2 classes.
# Using 16524 files for validation.
# Found 82621 files belonging to 2 classes.
# Epoch 1/15
# 2583/2583 [==============================] - 824s 318ms/step - loss: 0.5882 - accuracy: 0.6873
# Epoch 2/15
# 2583/2583 [==============================] - 838s 324ms/step - loss: 0.5066 - accuracy: 0.7434
# Epoch 3/15
# 2583/2583 [==============================] - 719s 278ms/step - loss: 0.4343 - accuracy: 0.7907
# Epoch 4/15
# 2583/2583 [==============================] - 759s 294ms/step - loss: 0.3635 - accuracy: 0.8322
# Epoch 5/15
# 2583/2583 [==============================] - 777s 301ms/step - loss: 0.2894 - accuracy: 0.8717
# Epoch 6/15
# 2583/2583 [==============================] - 761s 294ms/step - loss: 0.2134 - accuracy: 0.9099
# Epoch 7/15
# 2583/2583 [==============================] - 762s 295ms/step - loss: 0.1546 - accuracy: 0.9377
# Epoch 8/15
# 2583/2583 [==============================] - 764s 296ms/step - loss: 0.1181 - accuracy: 0.9530
# Epoch 9/15
# 2583/2583 [==============================] - 700s 271ms/step - loss: 0.0936 - accuracy: 0.9645
# Epoch 10/15
# 2583/2583 [==============================] - 591s 229ms/step - loss: 0.0826 - accuracy: 0.9700
# Epoch 11/15
# 2583/2583 [==============================] - 641s 248ms/step - loss: 0.0705 - accuracy: 0.9748
# Epoch 12/15
# 2583/2583 [==============================] - 609s 236ms/step - loss: 0.0629 - accuracy: 0.9774
# Epoch 13/15
# 2583/2583 [==============================] - 593s 229ms/step - loss: 0.0579 - accuracy: 0.9799
# Epoch 14/15
# 2583/2583 [==============================] - 624s 241ms/step - loss: 0.0544 - accuracy: 0.9812
# Epoch 15/15
# 2583/2583 [==============================] - 624s 242ms/step - loss: 0.0503 - accuracy: 0.9827
# 2582/2582 [==============================] - 206s 80ms/step - loss: 0.1150 - accuracy: 0.9637
# Test accuracy: 0.9636533260345459
#
# Process finished with exit code 0
