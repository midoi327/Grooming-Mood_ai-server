import graphlib
import json
#from pyexpat import model
from re import A, I
from flask import Flask, render_template, request, jsonify
from flask.json import JSONDecoder
import keras
import numpy as np
import cv2 as cv
from keras.models import Sequential
from keras.layers import  Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
import pandas as pd
from skimage import color
import tensorflow as tf

app = Flask(__name__)

#동영상 파일을 변수로 넣으면 60프레임마다 한번씩 캡쳐본을 생성해주는 함수
def videocapture(filepath): 
  input_video = cv.VideoCapture(filepath)
  count = 0
  while(input_video.isOpened()):
    ret, frame = input_video.read() #프레임 생성
    if not ret: 
      break
    if(int(input_video.get(1)) % 60 == 0): #프레임 60 당 이미지 1개 캡쳐
        frame = cv.resize(frame, (48, 48))
        cv.imwrite("/content/gdrive/MyDrive/CastoneDesign/FrameTest/4%d.jpg" % count, frame) #이미지 저장
        print('Saved frame %d.jpg' %count)
        count +=1
    
  input_video.release()
  return count #캡쳐본 이미지 개수 반환 

#이미지를 변수로 넣으면 가장 우세한 감정과 확률을 구해주는 함수
def predfunction(img): 
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


@app.route("/jsontest", methods=['POST']) #json 주고받기 테스트
def jsontest():
    result = request.get_json()
    return jsonify({result}), 200


@app.route("/predict_face", methods=['POST']) # prediction Api
def face_model():
    if request.method == 'POST':
        count = 0
        maxEmotion = {}
        init = 0

        file = request.files['video'] #동영상 파일 경로가 있어야함
        count = videocapture(file) #프레임 캡쳐 후 캡쳐 이미지 개수 반환

        for i in range(count):
            img = img = "/content/gdrive/MyDrive/CastoneDesign/FrameTest/4"+str(i)+".jpg" #캡쳐본 파일 경로
            pred = predfunction(img)

            if pred['prob'] > init:
                max_prob = pred['prob']
                max_index = pred['index']

        maxEmotion['index'] = max_index
        maxEmotion['prob'] = max_prob

    return jsonify(maxEmotion)



if __name__ == "__main__":

    app.run()