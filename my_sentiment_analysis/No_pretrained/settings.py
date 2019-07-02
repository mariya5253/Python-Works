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



def init():
    
	data = pd.read_csv('./data/BSLCMS-Comments.csv', usecols=['comments', 'comment_status'])# p_test.SentimentText=p_test.SentimentText.astype(str)
	# data =data.dropna()
	# print(data)
	# data.drop_duplicates(subset ="comments", keep = False, inplace = True) 
	data.columns = ['text', 'sentiment']
	data.text = data.text.astype(str)
	data = data.sample(frac=1, random_state=42)
	print(data.shape)

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

	# saving
	with open('tokenizer.pickle', 'wb') as handle:
	    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def tokenize(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r"#(\w+)", '', tweet)
    tweet = re.sub(r"@(\w+)", '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # tweet = re.sub(r'^[-+]?[0-9]*\.[0-9]+$', '', tweet)
    tweet = tweet.strip().lower()
    tokens = word_tokenize(tweet)
    return tokens


