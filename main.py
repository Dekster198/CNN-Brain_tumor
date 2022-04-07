import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import cv2
from google.colab import drive

drive.mount('/content/drive/')

DATADIR = '/content/drive/MyDrive/Datasets/Brain_tumor_dataset/'
CATEGORIES = ['no', 'yes']
IMG_SIZE = 500
training_data = []
x = []
y = []

def create_training_data():
  for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    num_class = CATEGORIES.index(category)
    for img in os.listdir(path):
      img_read = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
      img_arr = cv2.resize(img_read, (IMG_SIZE, IMG_SIZE))
      training_data.append([img_arr, num_class])

create_training_data()

random.shuffle(training_data)

for features, label in training_data:
  x.append(features)
  y.append(label)

x = np.array(x)
y = np.array(y)

x = x / 255
y = keras.utils.to_categorical(y, 2)

model = keras.Sequential([
                          Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
                          MaxPooling2D((2,2)),
                          Conv2D(64, (3,3), activation='relu'),
                          MaxPooling2D((2,2)),
                          Conv2D(128, (3,3), activation='relu'),
                          MaxPooling2D((2,2)),
                          Flatten(),
                          Dense(512, activation='relu'),
                          Dense(2, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=10, batch_size=16)

dataset_classes = {0:'No', 1:'Yes'}
k = 0

for i in range(100):
  n = np.random.randint(0, 253)
  ind = np.expand_dims(x[n], axis=0)
  res_ind = np.argmax(model.predict(ind))
  orig_ind = np.argmax(y[n])

  plt.imshow(x[n])
  plt.show()

  print('Распознанное значение: ', dataset_classes[res_ind], '\nОжидаемое значение: ', dataset_classes[orig_ind])

  if res_ind == orig_ind:
    k += 1

print('Количество распознанных изображение: ', k)
