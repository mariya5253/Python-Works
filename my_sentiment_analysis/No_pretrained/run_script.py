import os
import re
import sys
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


def tokenize(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r"#(\w+)", '', tweet)
    tweet = re.sub(r"@(\w+)", '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = tweet.strip().lower()
    tokens = word_tokenize(tweet)
    return tokens

tqdm.pandas()

my_model = load_model('./models/rnn_no_embeddings/weights-improvement-01-0.8000.hdf5')

tokenizer = Tokenizer()
# data['tokens'] = sys.argv[0](tokenize)
data = pd.DataFrame()
data['tokens'] = tokenize(sys.argv[0])
data['cleaned_text'] = data['tokens'].map(lambda tokens: ' '.join(tokens))
test_sequences = tokenizer.texts_to_sequences(data['cleaned_text'])

MAX_LENGTH = 35

padded_test_sequences = pad_sequences(test_sequences, maxlen=MAX_LENGTH)
prediction = my_model.predict(padded_test_sequences, verbose=1, batch_size=2048)
# y_pred_rnn_simple = pd.DataFrame(prediction, columns=['prediction'])
# y_pred_rnn_simple['prediction'] = y_pred_rnn_simple['prediction'].map(lambda p: 1 if p >= 0.5 else 0)
# print(y_pred_rnn_simple['prediction'])
print(prediction)

