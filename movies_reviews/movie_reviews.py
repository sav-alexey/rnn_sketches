import keras as k
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from sklearn.model_selection import train_test_split
import re
import io
import os


"""
Creating new data frame with two columns:
first column for movie reviews, and the second one for estimation
"""
df = pd.DataFrame(columns = ['review', 'sentiment'])
path = "txt_sentoken"
l = []

"""
Gathering all reviews from different folders into one csv file. 
Assigning "1" for positive reviews and "0" for negative reviews:
"""
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

df.to_csv("Data.csv") 
"""
Removing all characters that are not letters or numbers:
"""
df['review'] = df['review'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))   


"""
Defining our vocabulary with 1500 most common words. 
For each of this 1500 words will be assigned unique number. 
0 is a reserved index that won't be assigned to any word. 
Then shaping a sequences of tokens based on our vocabulary. 
Every word will be replaced according to their tokens. 
Words that are not in our vocabulary won't be included. 
Finally creating an array contained our sequences with padding
and dimension with number of timesteps.
"""   
tokenizer = k.preprocessing.text.Tokenizer(num_words=1500, split=' ')
tokenizer.fit_on_texts(df['review'].values)
X = tokenizer.texts_to_sequences(df['review'].values)
X = k.preprocessing.sequence.pad_sequences(X)

"""
Setting parameters:
"""
output_dim = 64
lstm_units = 100
dropout = 0.5
batch_size = 32
epochs = 3
optimizer = k.optimizers.Adam(lr=0.003)

"""
Setting our model:
"""
model = Sequential()
model.add(Embedding(2500, output_dim=output_dim,input_length = X.shape[1], dropout = dropout))
model.add(LSTM(lstm_units, dropout_U = dropout, dropout_W = dropout))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer=optimizer,metrics = ['accuracy'])

#Y = pd.get_dummies(df['sentiment']).values
Y = df['sentiment'].values

"""
Splitting our data into random train and test subsets:
"""
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)

#model = k.models.load_model('saved_model_sigmoid.h5')
"""
Fitting and saving model:
"""
model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs)
model.save('saved_model_sigmoid.h5')


#with io.open("X_test.txt", "r") as f:
#        X_test = f.read().lower()

"""
Making predictions using test subsets:
"""
prediction = model.predict(X_test)
prediction = np.where(prediction<0.5, 0, 1)

accuracy = k.metrics.binary_accuracy(prediction.flatten(), Y_test.astype(int))
print(k.backend.eval(accuracy))

#testlist = []
#testlist.append(X_test)

#print(testlist)
#X_test = tokenizer.texts_to_sequences(testlist)
#maxlen = X.shape[1]
#X_test = k.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)
#print(X_test[:,:])

#pretrained_model = k.models.load_model('saved_model_sigmoid.h5')
#prediction = pretrained_model.predict(X_test)
#sentiment = np.where(prediction[0][0]<0.5, "positive", "negative")
#print(prediction)
#print(prediction, sentiment)
    















