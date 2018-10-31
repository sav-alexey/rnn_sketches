import keras as k
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from sklearn.model_selection import train_test_split
import re
import io
import os
import pandas as pd

df = pd.DataFrame(columns = ['review', 'sentiment'])
path = "C:/Users/Administrator/.spyder-py3/Git/rnn_sketches/txt_sentoken"
l = []
pos_reviews = os.listdir(path + '/pos/')
for i in range(len(pos_reviews)):
    with io.open(path+'/pos/'+pos_reviews[i], "r") as f:
        text = f.read().lower()
        df = df.append({'review':text, 'sentiment': 1}, ignore_index=True)
        
neg_reviews = os.listdir(path + '/neg/')
for i in range(len(pos_reviews)):
    with io.open(path+'/neg/'+neg_reviews[i], "r") as f:
        text = f.read().lower()
        df = df.append({'review':text, 'sentiment': 0}, ignore_index=True)

df['review'] = df['review'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))   
     
tokenizer = k.preprocessing.text.Tokenizer(num_words=2500, split=' ')
tokenizer.fit_on_texts(df['review'].values)
X = tokenizer.texts_to_sequences(df['review'].values)
X = k.preprocessing.sequence.pad_sequences(X)
#print(tokenizer.word_index) 
#embed_dim = 128
#lstm_out = 200
#batch_size = 32
#
#model = Sequential()
#model.add(Embedding(2500, embed_dim,input_length = X.shape[1], dropout = 0.2))
#model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
#model.add(Dense(2,activation='softmax'))
#model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
#
#Y = pd.get_dummies(df['sentiment']).values
#X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.20, random_state = 36)
#
#model.fit(X_train, Y_train, batch_size = batch_size, nb_epoch = 1,  verbose = 1)
#model.save('saved_model.h5')

with io.open("X_test.txt", "r") as f:
        X_test = f.read().lower()
#
df = df.append({'review':X_test}, ignore_index=True) 
print(df['review'].values[-1])  

X_test = tokenizer.texts_to_sequences(df['review'].values)[-1]
print(X_test)
#
#pretrained_model = k.models.load_model('saved_model.h5')
#prediction = pretrained_model.predict(X_test)
#print(prediction)
    