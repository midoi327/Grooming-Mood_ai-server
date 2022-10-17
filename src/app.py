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
from werkzeug.utils import secure_filename
import soundfile as sf
import wave, array
import joblib
import librosa
from tqdm import tqdm
import os
import moviepy.editor as mp
from pydub import AudioSegment
import pydub
from os.path import join

pydub.AudioSegment.converter= "C:/ffmpeg/ffmpeg/bin/ffmpeg.exe"
app = Flask(__name__)
#---------------------------------------- 표정 인식에 필요한 함수 -------------------------------------------------------------------------

#표정 인식: 동영상 파일을 변수로 넣으면 60프레임마다 한번씩 캡쳐본을 생성해주는 함수
def videocapture(filepath): 
  input_video = cv.VideoCapture(filepath)
  count = 0
  while(input_video.isOpened()):
    ret, frame = input_video.read() #프레임 생성
    if not ret: 
      break
    if(int(input_video.get(1)) % 60 == 0): #프레임 60 당 이미지 1개 캡쳐
        frame = cv.resize(frame, (48, 48))
        cv.imwrite("C:/Users/imreo/gromming-mood-flask/src/dataset/FrameTest/%d.jpg" % count, frame) #캡쳐본 로컬에 저장
        print('Saved frame %d.jpg' %count)
        count +=1
    
  input_video.release()
  return count #캡쳐본 이미지 개수 반환 

#표정 인식: 이미지를 변수로 넣으면 가장 우세한 감정과 확률을 구해주는 함수
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


#---------------------------------------- 음성 인식에 필요한 함수 -------------------------------------------------------------------------

#음성 인식: 영상(mp4)로부터 음성(wav)를 추출하는 함수
def extract_wav_from_mp4(file_name):

    file = mp.VideoFileClip(file_name)
    file.audio.write_audiofile("src/dataset/AudioMp3.mp3") #mp4를 mp3로 추출
    
    src = "C:/Users/imreo/gromming-mood-flask/src/dataset/AudioMp3.mp3"
    dst="src/dataset/AudioWav.wav"

    audSeg = pydub.AudioSegment.from_mp3(src)
    audSeg.export("src/dataset/AudioWav.wav", format="wav") #mp3를 wav로 변환
    
    wavefilepath = "src/dataset/AudioWav.wav"

    return wavefilepath

#음성 인식: 음성(wav)으로부터 mfcc 특징벡터를 추출하는 함수
def extract_mfcc_feature(file_name):
    audio_signal, sample_rate = librosa.load(file_name, sr=22050)
    spectrogram = librosa.stft(audio_signal, n_fft=512)
    spectrogram = np.abs(spectrogram)

    power_spectrogram = spectrogram**2
    mel = librosa.feature.melspectrogram(S=power_spectrogram, sr=sample_rate)
    mel = librosa.power_to_db(mel)

    mfccs = librosa.feature.mfcc(S=mel, n_mfcc=20)
    mfcc_feature = np.mean(mfccs.T, axis=0)

    return mfcc_feature

#음성 인식: mfcc로부터 감정을 예측하는 함수
def pred_voice_emotion(mfccs):
    x_test = np.array(mfccs)
    clf = joblib.load("C:/Users/imreo/gromming-mood-flask/src/model_test.pkl")
    predict = clf.predict(x_test)
    print("predict: ", predict)
    
    UserEmotion = [] #유저의 감정 결과 저장할 list

    #5. 유저의 감정 4가지로 변환 (happy, neutral, sad, angry)
    for i in predict:
        if i == 'angry': UserEmotion.append(3)
        elif i == 'calm': UserEmotion.append(1)
        elif i == 'disgust': UserEmotion.append(3)
        elif i == 'fearful' : UserEmotion.append(1)
        elif i == 'happy' : UserEmotion.append(0)
        elif i == 'neutral' : UserEmotion.append(1)
        elif i == 'sad' : UserEmotion.append(2)
        elif i == 'surprised' : UserEmotion.append(1)
    
    return UserEmotion # ex) [0,1,0,3,2] 




#----------------------------------------표정 인식 API -------------------------------------------------------------------------

#표정으로부터 감정 인식 API
@app.route("/predict_face", methods=['POST']) 
def face_model():
    if request.method == 'POST':
        count = 0
        maxEmotion = {}
        init = 0

        file = request.files['video'] #동영상 파일 경로가 있어야함
        file.save('src/dataset/'+secure_filename(file.filename)) #동영상을 로컬에 저장

        videofile = "C:/Users/imreo/gromming-mood-flask/src/dataset/" + file.filename #캡처할 비디오 파일 경로

        count = videocapture(videofile) #프레임 캡쳐 후 캡쳐 이미지 개수 반환
        
        for i in range(count):
            img = img = "C:/Users/imreo/gromming-mood-flask/src/dataset/FrameTest/"+str(i)+".jpg" #캡쳐본 파일 경로
            pred = predfunction(img) #캡쳐본의 감정 인식하기

            if pred['prob'] > init: #가장 강렬한 감정 선택
                max_prob = pred['prob']
                max_index = pred['index']

        maxEmotion['Emotion'] = max_index
        maxEmotion['Probability'] = float(max_prob)
        print(maxEmotion)

    return jsonify(maxEmotion), 200



#----------------------------------------음성 인식 API -------------------------------------------------------------------------
#음성으로부터 감정 인식 API
@app.route("/predict_voice", methods=['POST'])
def voice_model():

  file = request.files['video'] #동영상 파일 경로가 있어야함
  file.save('src/dataset/'+secure_filename(file.filename)) #동영상을 로컬에 저장

  #영상 경로
  videofile = "C:/Users/imreo/gromming-mood-flask/src/dataset/" + file.filename 

  #1. 음성 파일 load
  wavepath = extract_wav_from_mp4(videofile)
  w = wave.open(wavepath) 
  wavLen = w.getnframes() / w.getframerate()
  print("음성의 길이는",wavLen,"초 입니다.")
  wavLenInt = int(wavLen)
  y, sr = librosa.load(wavepath, sr=22050) 

  #2. 음성 파일 4초씩 자르기
  count = 1
  for i in range(0, wavLenInt, 4):
      if((i+4)>wavLen) : break
      cutted_y = y[i*22050:(i+4)*22050] #(0~4). (4~8). (8~12)초씩 저장
      sf.write("C:/Users/imreo/gromming-mood-flask/src/dataset/WaveTest/"+"cutted_"+ str(count)+ '.wav', cutted_y, sr, 'PCM_16')
      count +=1 


  #3. 자른 음성 파일들 mfcc 특징 벡터 구하기
  folder = "C:/Users/imreo/gromming-mood-flask/src/dataset/WaveTest/"
  files_list = os.listdir(folder)
  print(files_list)

  mfccs = []

  for file_name in tqdm(files_list):
    mfccs.append(extract_mfcc_feature(join(folder,file_name)))
  
  #4. 분류기로 감정 예측하기
  UserEmotion = pred_voice_emotion(mfccs)
  
  #5. 유저의 감정 중 가장 우세한 감정 구하기
  CntHappy = UserEmotion.count(0)
  CntNeutral = UserEmotion.count(1)
  CntSad = UserEmotion.count(2)
  CntAngry = UserEmotion.count(3)

  UserMaxEmotion = [CntHappy, CntNeutral, CntSad, CntAngry] #각각 감정 개수

  result = {}
  Emotion = np.argmax(UserMaxEmotion) #가장 우세한 감정

  result['Emotion'] = int(Emotion) #결과 전달
  print(result)
  

  return jsonify(result), 200




#---------------------------------------flask api 실행 --------------------------------------------------------------------
if __name__ == "__main__":

    app.run()