
import numpy as np 
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import cv2 as cv
from skimage import color


#동영상 파일을 변수로 넣으면 60프레임마다 한번씩 캡쳐본을 생성해주는 함수
def videocapture(filepath): 
  print("videocapture 진입")
  input_video = cv.VideoCapture(filepath)
  count = 0
  while(input_video.isOpened()):
    ret, frame = input_video.read() #프레임 생성
    if not ret: 
      break
    if(int(input_video.get(1)) % 60 == 0): #프레임 60 당 이미지 1개 캡쳐
        frame = cv.resize(frame, (48, 48))
        cv.imwrite("C:/Users/imreo/gromming-mood-flask/src/dataset/FrameTest/%d.jpg" % count, frame) #이미지 저장
        print('Saved frame %d.jpg' %count)
        count +=1
    
  input_video.release()
  print("count는",count)
  return count #캡쳐본 이미지 개수 반환 

#이미지를 변수로 넣으면 가장 우세한 감정과 확률을 구해주는 함수
def predfunction(img): 
  print("predfunction 진입")
  result = {}
  img = cv.imread(img)
  img = color.rgb2gray(img)
  img = img.reshape(-1, 48, 48, 1)

  model = tf.keras.models.load_model('C:/Users/imreo/gromming-mood-flask/src/cpu_face_model.h5')
  prediction = model.predict(img)

  result['prob'] = max(max(prediction))
  index = np.argmax(prediction)

  if (index == 0): result['index'] = 3
  elif (index == 1): result['index'] = 3
  elif (index == 2): result['index'] = 1
  elif (index == 3) : result['index'] = 0
  elif (index == 4) : result['index'] = 2
  elif (index == 5) : result['index'] = 1
  elif (index == 6): result['index'] = 1

  return result # 0: happy, 1: neutral, 2: sad, 3: angry


count = 0
maxEmotion = {}
init = 0

file = "C:/Users/imreo/gromming-mood-flask/src/dataset/happy7.mp4"  #동영상 파일 경로가 있어야함
count = videocapture(file) #프레임 캡쳐 후 캡쳐 이미지 개수 반환

for i in range(count):
    img = "C:/Users/imreo/gromming-mood-flask/src/dataset/FrameTest/"+str(i)+".jpg" #캡쳐본 파일 경로
    pred = predfunction(img)

    if pred['prob'] > init:
        max_prob = pred['prob']
        max_index = pred['index']

maxEmotion['Emotion'] = max_index
maxEmotion['Probability'] = max_prob


print("감정 분석 결과: %d번째 감정이 %.2f 확률로 가장 우세합니다." %(maxEmotion['Emotion'],maxEmotion['Probability']))