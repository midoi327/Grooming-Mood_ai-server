import os
import joblib
import librosa
import sklearn
import soundfile as sf
import numpy as np
import wave, array
import pickle
from imutils import paths
from os.path import join
from tqdm import tqdm
import moviepy.editor as mp
from pydub import AudioSegment
import pydub
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pydub.AudioSegment.converter= "C:/ffmpeg/ffmpeg/bin/ffmpeg.exe"

filepath = "C:/Users/imreo/gromming-mood-flask/src/dataset/HappyFace.mp4" #받아온 웹캠 영상

#mp4 파일로부터 wav 추출하는 함수
def extract_wav_from_mp4(file_name):

    file = mp.VideoFileClip(file_name)
    file.audio.write_audiofile("src/dataset/AudioMp3.mp3") #mp4를 mp3로 추출
    
    src = "C:/Users/imreo/gromming-mood-flask/src/dataset/AudioMp3.mp3"
    dst="src/dataset/AudioWav.wav"

    audSeg = pydub.AudioSegment.from_mp3(src)
    audSeg.export("src/dataset/AudioWav.wav", format="wav") #mp3를 wav로 변환
    
    wavefilepath = "src/dataset/AudioWav.wav"

    return wavefilepath


#wav 파일로부터 mfcc 특징벡터를 추출하는 함수
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


#mfcc로부터 감정을 예측하는 함수
def pred_voice_emotion(mfccs, count):
    x_test = np.array(mfccs)
    clf = joblib.load("C:/Users/imreo/gromming-mood-flask/src/voice_svm_model.pkl")
    
    predict = []
    maxProb = 0
    userEmotion = 1

    scaler = MinMaxScaler()
    scaler.fit(mfccs)
    mfccs_scaled = scaler.transform(mfccs)

    predict = clf.predict_proba(mfccs_scaled) 
    print(x_test)
    print(predict)
    #라벨 ( angry, calm, disgust, fearful, happy, neutral, sad, surprised)
    for i in range(0, count-1):
        pred = predict[i] #i번째 파일
        for w in range(0,7): #0~7 감정 확률 
            prob = pred[w] # i번째 음성파일 w번째 음성
            if prob > maxProb:
                maxProb = prob
                userEmotion = w #가장 강렬한 감정

    #유저의 감정 4가지로 변환 (happy, neutral, sad, angry)

    if userEmotion == 0: UserEmotion = 3
    elif userEmotion == 1: UserEmotion = 1
    elif userEmotion == 2: UserEmotion= 3
    elif userEmotion == 3 : UserEmotion= 1
    elif userEmotion == 4 : UserEmotion= 0
    elif userEmotion == 5 : UserEmotion= 1
    elif userEmotion == 6 : UserEmotion = 2
    elif userEmotion == 7 : UserEmotion= 1
    
    print("UseEmotion, maxProb : %d, %.3f" %(UserEmotion,maxProb))



    

    return [UserEmotion, maxProb]




#1. 음성 파일 load
wavepath = extract_wav_from_mp4(filepath)
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
[UserEmotion, maxProb] = pred_voice_emotion(mfccs, count)



