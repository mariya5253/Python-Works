import os
import re

import warnings
warnings.simplefilter("ignore", UserWarning)

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np 
from string import punctuation

from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.externals import joblib

import scipy
from scipy.sparse import hstack
from tqdm import tqdm, tqdm_notebook
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras.models import Sequential

from keras.layers import Input, Dense, Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.layers import SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.callbacks import Callback
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.utils.vis_utils import plot_model

tqdm.pandas()
data = pd.read_csv('./data/Sentiment Analysis Dataset.csv', encoding='latin1', usecols=['Sentiment', 'SentimentText'])
# p_test.SentimentText=p_test.SentimentText.astype(str)
# data =data.dropna()
print(data)
data.columns = ['sentiment', 'text']
data.text = data.text.astype(str)
data = data.sample(frac=1, random_state=42)
print(data.shape)

def tokenize(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r"#(\w+)", '', tweet)
    tweet = re.sub(r"@(\w+)", '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = tweet.strip().lower()
    tokens = word_tokenize(tweet)
    return tokens



data['tokens'] = data.text.progress_map(tokenize)
data['cleaned_text'] = data['tokens'].map(lambda tokens: ' '.join(tokens))
data[['sentiment', 'cleaned_text']].to_csv('./data/cleaned_text.csv')

data = pd.read_csv('./data/cleaned_text.csv')
print(data.shape)
data =data.dropna()

x_train, x_test, y_train, y_test = train_test_split(data['cleaned_text'], 
                                                    data['sentiment'], 
                                                    test_size=0.1, 
                                                    random_state=42,
                                                    stratify=data['sentiment'])

pd.DataFrame(y_test).to_csv('./predictions/y_true.csv', index=False, encoding='utf-8')
# x_train = x_train.dropna()
# print(x_train)
MAX_NB_WORDS = 80000
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(str(data['cleaned_text']))


train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

MAX_LENGTH = 35
padded_train_sequences = pad_sequences(train_sequences, maxlen=MAX_LENGTH)
padded_test_sequences = pad_sequences(test_sequences, maxlen=MAX_LENGTH)

def get_simple_rnn_model():
    embedding_dim = 300
    embedding_matrix = np.random.random((MAX_NB_WORDS, embedding_dim))
    
    inp = Input(shape=(MAX_LENGTH, ))
    x = Embedding(input_dim=MAX_NB_WORDS, output_dim=embedding_dim, input_length=MAX_LENGTH, 
                  weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(100, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

rnn_simple_model = get_simple_rnn_model()


filepath="./models/rnn_no_embeddings/weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

batch_size = 256
epochs = 2

history = rnn_simple_model.fit(x=padded_train_sequences, 
                    y=y_train, 
                    validation_data=(padded_test_sequences, y_test), 
                    batch_size=batch_size, 
                    callbacks=[checkpoint], 
                    epochs=epochs, 
                    verbose=1)

best_rnn_simple_model = load_model('./models/rnn_no_embeddings/weights-improvement-01-0.8000.hdf5')

y_pred_rnn_simple = best_rnn_simple_model.predict(padded_test_sequences, verbose=1, batch_size=2048)

# y_pred_rnn_simple = pd.DataFrame(y_pred_rnn_simple, columns=['prediction'])
# y_pred_rnn_simple['prediction'] = y_pred_rnn_simple['prediction'].map(lambda p: 1 if p >= 0.5 else 0)
# y_pred_rnn_simple.to_csv('./predictions/y_pred_rnn_simple.csv', index=False)
# y_pred_rnn_simple = pd.read_csv('./predictions/y_pred_rnn_simple.csv')
# print(accuracy_score(y_test, y_pred_rnn_simple))

print(y_pred_rnn_simple)