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

filepath = "src/dataset/WaveTest/audioTest.wav"

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




#음성 파일 4초씩 잘라서 저장하기 
w = wave.open(filepath)
wavLen = w.getnframes() / w.getframerate()
print("음성의 길이는",wavLen,"초 입니다.")
wavLenInt = int(wavLen)

y, sr = librosa.load(filepath, sr=22050) #음성 읽기

count = 1
for i in range(0, wavLenInt, 4):
    if((i+4)>wavLen) : break
        
    cutted_y = y[i*22050:(i+4)*22050] #(0~4). (4~8). (8~12)초씩 저장
    
    sf.write("C:/Users/imreo/face_sentiment_flask/src/dataset/WaveTest/"+"cutted_"+ str(count)+ '.wav', cutted_y, sr, 'PCM_16')
    
    count +=1 



#자른 음성 파일들 mfcc 특징 벡터 구하기
folder = "C:/Users/imreo/face_sentiment_flask/src/dataset/WaveTest/"
files_list = os.listdir(folder)
print(files_list)

mfccs = []

for file_name in tqdm(files_list):
    mfccs.append(extract_mfcc_feature(join(folder,file_name)))


#감정 예측하기
x_test = np.array(mfccs)
clf = joblib.load("C:/Users/imreo/face_sentiment_flask/src/model_test.pkl")
predict = clf.predict(x_test)


print("predict: ", predict)

UserEmotion = [] #유저의 감정 결과 저장할 list

#유저의 감정 4가지로 변환 (happy, neutral, sad, angry)
for i in predict:
    if i == 'angry': UserEmotion.append(3)
    elif i == 'calm': UserEmotion.append(1)
    elif i == 'disgust': UserEmotion.append(3)
    elif i == 'fearful' : UserEmotion.append(1)
    elif i == 'happy' : UserEmotion.append(0)
    elif i == 'neutral' : UserEmotion.append(1)
    elif i == 'sad' : UserEmotion.append(2)
    elif i == 'surprised' : UserEmotion.append(1)


#유저의 감정 중 가장 우세한 감정 구하기
CntHappy = UserEmotion.count(0)
CntNeutral = UserEmotion.count(1)
CntSad = UserEmotion.count(2)
CntAngry = UserEmotion.count(3)
print("happy, neutral, sad, angry 개수: %d, %d, %d, %d" %(CntHappy, CntNeutral, CntSad, CntAngry))

UserMaxEmotion = [CntHappy, CntNeutral, CntSad, CntAngry] #각각 감정 개수
maxEmotion = np.argmax(UserMaxEmotion) #가장 우세한 감정

print("음성 감정 분석 결과: 유저의 가장 우세한 감정은 %d번째 감정입니다." %maxEmotion)
