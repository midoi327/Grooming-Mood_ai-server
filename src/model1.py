import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import cv2 as cv
from skimage import color
import keras


train_csv = pd.read_csv("C:/Users/imreo/face_sentiment_flask/src/dataset/train.csv", encoding='utf-8')

with open("C:/Users/imreo/face_sentiment_flask/src/dataset/train.csv") as f:
    content = f.readlines()
    lines_train = np.array(content)
    num_of_train_instances = lines_train.size


#데이터 전처리
x_train, y_train, x_test, y_test = [], [] ,[], []

for i in range(1, num_of_train_instances):
    try:
        emotion, img = lines_train[i].split(",")
        img = img.replace('"', '')
        val = img.split(" ")
        pixels = np.array(val, 'float32')
        
        emotion = tf.keras.utils.to_categorical(emotion, num_classes = 7)
        
        y_train.append(emotion)
        x_train.append(pixels)
    except:
        print("this is over", end="")

#여기에 train 데이터의 일부를 test 데이터로 사용하기 위해서 나누자
x_train, y_train = x_train[5000:], y_train[5000:]
x_test, y_test = x_train[:5000], y_train[:5000]

emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

#데이터 타입 list에서 array로 변경

x_train = np.array(x_train, 'float32')/255
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')/255
y_test = np.array(y_test, 'float32')

#차원 수 4차원으로 변경 
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)

#모델 만들기
model = tf.keras.models.Sequential()   

model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape =(48,48,1)))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='Adam',
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
model.summary()


#모델 학습을 위해 np 데이터를 tensor 데이터로 변환
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
x_test= tf.convert_to_tensor(x_test, dtype=tf.float32)

#모델 학습
model.fit(x_train, y_train, epochs=20, verbose=2)

#모델 평가하기
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])

#모델 저장
model.save('cpu_face_model.h5')
