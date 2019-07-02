import os
import re

import warnings
warnings.simplefilter("ignore", UserWarning)
from matplotlib import pyplot as plt
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
# from tqdm import tqdm_notebook

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
import math
import scipy
from scipy.sparse import hstack
from tqdm import tqdm, tqdm_notebook
#tqdm.pandas(desc="progress-bar")
tqdm.pandas()
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
import pickle



data = pd.read_csv('./data/Sentiment Analysis Dataset.csv',encoding='utf-8', usecols=['Sentiment', 'SentimentText'])
# data =data.dropna()
data.columns = ['sentiment', 'text']
data = data.sample(frac=1, random_state=42)
data.text = data.text.astype(str)
print(data.shape)

def tokenize(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r"#(\w+)", '', tweet)
    tweet = re.sub(r"@(\w+)", '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # tweet = re.sub(r'^[-+]?[0-9]*\.[0-9]+$', '', tweet)
    tweet = tweet.strip().lower()
    tokens = word_tokenize(tweet)
    return tokens



data['tokens'] = data.text.progress_map(tokenize)
data['cleaned_text'] = data['tokens'].map(lambda tokens: ''.join(tokens))
data[['sentiment', 'cleaned_text']].to_csv('./data/cleaned_text.csv')

data = pd.read_csv('./data/cleaned_text.csv')
print(data.shape)

data =data.dropna()
x_train, x_test, y_train, y_test = train_test_split(data['cleaned_text'], 
                                                    data['sentiment'], 
                                                    test_size=0.1, 
                                                    random_state=42,
                                                    stratify=data['sentiment'])

# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(x_test)

pd.DataFrame(y_test).to_csv('./predictions/y_true.csv', index=False, encoding='utf-8')



def get_coefs(word, *arr):
    try:
        # print("word:",word)
        # print("arr:",arr)
        return word, np.asarray(arr, dtype='float32')
    except:
        return None, None
    
embeddings_index = dict(get_coefs(*o.strip().split()) for o in tqdm_notebook(open('./embeddings/glove.twitter.27B.50d.txt',encoding="utf8")))

#print(embeddings_index)
 
embed_size=50

for k in tqdm_notebook(list(embeddings_index.keys())):
    v = embeddings_index[k]
     
    try:
        if v.shape != (embed_size, ):
            embeddings_index.pop(k)
            i =i + 1
    except:
        pass
            
#embeddings_index.pop(None)

values = list(embeddings_index.values())

all_embs = np.stack(values)

emb_mean, emb_std = all_embs.mean(), all_embs.std()

MAX_NB_WORDS = 80000
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

# tokenizer.fit_on_texts(data['cleaned_text'])
try:
   tokenizer.fit_on_texts(data['cleaned_text'].values)
except Exception as e:
   print("exceiption is", e)
   print("data passedin ", data['cleaned_text'].values)

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
word_index = tokenizer.word_index
nb_words = MAX_NB_WORDS
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

oov = 0
for word, i in word_index.items():
    if i >= MAX_NB_WORDS: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        oov += 1

print(oov)

print(x_train)
# if math.isnan(x_train)!= True:
# x_train = np.nan_to_num(x_train)

# if np.isnan(np.x_train):
train_sequences = tokenizer.texts_to_sequences(x_train)

    
test_sequences = tokenizer.texts_to_sequences(x_test)

MAX_LENGTH = 35
padded_train_sequences = pad_sequences(train_sequences, maxlen=MAX_LENGTH)
padded_test_sequences = pad_sequences(test_sequences, maxlen=MAX_LENGTH)


def get_rnn_model_with_glove_embeddings():
    embedding_dim = 50
    inp = Input(shape=(MAX_LENGTH, ))
    x = Embedding(MAX_NB_WORDS, embedding_dim, weights=[embedding_matrix], input_length=MAX_LENGTH, trainable=True)(inp)
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

rnn_model_with_embeddings = get_rnn_model_with_glove_embeddings()

filepath="./models/rnn_with_embeddings/weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# rnn_model_with_embeddings.save_weights('./models/rnn_with_embeddings/my_model_weights.h5')
batch_size = 256
epochs = 4

history = rnn_model_with_embeddings.fit(x=padded_train_sequences, 
                    y=y_train, 
                    validation_data=(padded_test_sequences, y_test), 
                    batch_size=batch_size, 
                    callbacks=[checkpoint], 
                    epochs=epochs, 
                    verbose=1)

# my_model = load_model('./models/rnn_with_embeddings/weights-improvement-04-0.8346.hdf5')
# loss, acc = my_model.evaluate(padded_test_sequences, y_test, verbose=0)
# print('Test Accuracy: %f' % (acc*100))
# print(filepath)
best_rnn_model_with_glove_embeddings = load_model('./models/rnn_with_embeddings/weights-improvement-04-0.8346.hdf5')

y_pred_rnn_with_glove_embeddings = best_rnn_model_with_glove_embeddings.predict(
    padded_test_sequences, verbose=1, batch_size=2048)

y_pred_rnn_with_glove_embeddings = pd.DataFrame(y_pred_rnn_with_glove_embeddings, columns=['prediction'])
y_pred_rnn_with_glove_embeddings['prediction'] = y_pred_rnn_with_glove_embeddings['prediction'].map(lambda p: 
                                                                                                    1 if p >= 0.5 else 0)
y_pred_rnn_with_glove_embeddings.to_csv('./predictions/y_pred_rnn_with_glove_embeddings.csv', index=False)
y_pred_rnn_with_glove_embeddings = pd.read_csv('./predictions/y_pred_rnn_with_glove_embeddings.csv')
print(accuracy_score(y_test, y_pred_rnn_with_glove_embeddings))