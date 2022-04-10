import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import cv2
from google.colab import drive, files
from io import BytesIO
from PIL import Image

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
      img_read = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
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

x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x, y)

model = keras.Sequential([
                          Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
                          MaxPooling2D((2,2)),
                          Conv2D(64, (3,3), activation='relu'),
                          MaxPooling2D((2,2)),
                          Conv2D(128, (3,3), activation='relu'),
                          MaxPooling2D((2,2)),
                          Flatten(),
                          Dense(2, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train_split, y_train_split, epochs=10, batch_size=16, validation_data=(x_val_split, y_val_split))

print(model.summary())

dataset_classes = {0:'No', 1:'Yes'}
k_train = 0
k_val = 0

print('Обучающая выборка:')
for i in range(100):
  n = np.random.randint(0, len(x_train_split)-1)
  ind = np.expand_dims(x_train_split[n], axis=0)
  res_ind = np.argmax(model.predict(ind))
  orig_ind = np.argmax(y_train_split[n])

  plt.imshow(x_train_split[n], cmap=plt.cm.binary)
  plt.show()

  print('Распознанное значение: ', dataset_classes[res_ind], '\nОжидаемое значение: ', dataset_classes[orig_ind])

  if res_ind == orig_ind:
    k_train += 1

print('Выборка валидации')
for i in range(len(x_val_split)):
  ind = np.expand_dims(x_val_split[i], axis=0)
  res_ind = np.argmax(model.predict(ind))
  orig_ind = np.argmax(y_val_split[i])

  plt.imshow(x_val_split[i], cmap=plt.cm.binary)
  plt.show()

  print('Распознанное значение: ', dataset_classes[res_ind], '\nОжидаемое значение: ', dataset_classes[orig_ind])

  if res_ind == orig_ind:
    k_val += 1

print('Количество распознанных изображение в обучающей выборке: ', k_train, ' из ', 100)
print('Количество распознанных изображений в выборке валидации: ', k_val, ' из ', len(x_val_split))

img = Image.open(BytesIO(uploaded['img.jpg']))
plt.imshow(img)

img = np.array(img)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
x = img / 255
x = np.expand_dims(x, axis=0)

res = np.argmax(model.predict(x))
print(res)

if res == 0:
  print('No')
else:
  print('Yes')