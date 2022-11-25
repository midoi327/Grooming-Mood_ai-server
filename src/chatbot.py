import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer('jhgan/ko-sroberta-multitask')
sentences = [ "한국어 문장 임베딩을 위한 버트 모델입니다."]
embeddings = model.encode(sentences)

df = pd.read_csv('src/wellness_dataset.csv')
df = df[~df['챗봇'].isna()]

#df['embedding'] = pd.Series([[]]*len(df))
#print("pd.series끝")
print("전")
df['embedding'] = df['유저'].map(lambda x: list(model.encode(x)))
print("후")

df.to_csv('afterembedding.csv', index=False)

while(True):
    text = input("입력 문구를 입력해주세요:")
    embedding = model.encode(text)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    print(df.head())
    answer = df.loc[df['distance'].idxmax()]

    print('구분', answer['구분']) 
    print('유사한 질문', answer['유저'])
    print('챗봇 답변', answer['챗봇']) # 챗봇 적용 시 이 것만 필요
    print('유사도', answer['distance']) 