from flask import Flask, render_template, request
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
from tqdm import tqdm, tqdm_notebook
#tqdm.pandas(desc="progress-bar")
tqdm.pandas()
import scipy
from scipy.sparse import hstack
from tqdm import tqdm, tqdm_notebook
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras.models import Sequential,  model_from_json
# from sentiment_predictor import token_fit_function
import sys
import pickle
# modulename = 'settings'
# if modulename not in sys.modules:
#     print('You have not imported the  module')
#     import settings

# import importlib
# spam_spec = importlib.util.find_spec("settings")
# # print(spam_spec)
# if spam_spec is not None:
# 	mode = sys.modules['settings']

# import settings
from keras.layers import Input, Dense, Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.layers import SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.callbacks import Callback
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import pymysql

app = Flask(__name__)

# define a graph
graph1 = tf.Graph()

# cursorObject        = connectionObject.cursor()

def tokenize(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r"#(\w+)", '', tweet)
    tweet = re.sub(r"@(\w+)", '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = tweet.strip().lower()
    tokens = word_tokenize(tweet)
    return tweet

app = Flask(__name__)

@app.route('/')
def query():
   return render_template('query_form.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
	tqdm.pandas()
	# tf.keras.backend.clear_session()
	# Connect to the database
	connection = pymysql.connect(host='localhost',
	                             user='root',
	                             password='root',
	                             db='mariya',
	                             charset='utf8mb4',
	                             cursorclass=pymysql.cursors.DictCursor)
	with graph1.as_default():

		# def DemoNet():

		#     # a = tf.keras.layers.Input(shape=(128,))
		#     # b = tf.keras.layers.Dense(128)(a)
		#     # b1 = tf.keras.layers.Dense(128)(b)
		#     # b2 = tf.keras.layers.Dense(128)(b1)
		#     # b3 = tf.keras.layers.Dense(128)(b2)

		#     # model = tf.keras.models.Model(inputs=a, outputs=b2)
		#     # return model

		# 	visible = Input(shape=(400, ) )
		# 	# flatten = Flatten()(visible)
		# 	# batch = BatchNormalization()(flatten)
		# 	hidden1 = Dense(1, activation='relu')(visible)
		# 	hidden2 = Dropout(0.5)(hidden1)
		# 	output = Dense(2, activation='softmax')(hidden2)
		# 	model = Model(inputs=visible, outputs=output)
		# 	return model
		# settings.init()
		my_model = load_model('./models/rnn_with_embeddings/weights-improvement-04-0.9171.hdf5')

		data = pd.DataFrame()

		if request.method == 'POST':
			result = request.form
			
			comment=request.form['Comment']
			
			data['text'] = [comment]
			data['tokens'] = data.text.progress_map(tokenize)
			data['cleaned_text'] = data['tokens'].map(lambda tokens: ''.join(tokens))

			# loading
			with open('tokenizer.pickle', 'rb')as handle:
				tokenizer = pickle.load(handle)

			test_sequences = tokenizer.texts_to_sequences(data['cleaned_text'])

			MAX_LENGTH = 35 
			padded_test_sequences = pad_sequences(test_sequences, maxlen=MAX_LENGTH)

			prediction = my_model.predict(padded_test_sequences, verbose=1, batch_size=2048)
			print(prediction)
			y_pred_rnn_simple = pd.DataFrame(prediction, columns=['prediction'])
			print(y_pred_rnn_simple['prediction'])
			y_pred_rnn_simple['prediction'] = y_pred_rnn_simple['prediction'].map(lambda p: 1 if p >= 0.5 else 0)
			print(y_pred_rnn_simple['prediction'][0])
			
			
			with connection.cursor() as cursor:
				# Read a single record
				# print(comment)
				# print(prediction[0][0])
				# pred = prediction.map(lambda p: 1 if p >= 0.5 else 0)
				# print(pred)
				pred = y_pred_rnn_simple['prediction'][0].item()
				sql = "INSERT INTO comments(sentiment,sentiment_text) VALUES (%s, %s)"
				print(comment)
				print(pred)
				cursor.execute(sql, (pred,comment))

				sql1 = "SELECT * FROM comments ORDER BY id DESC"
				cursor.execute(sql1)
				# print(rows)
				# for row in rows:
				# 	print(row)
				rows = cursor.fetchall()
				# for row in rows:
				# 	print(row['sentiment'])
				# 	print(row['sentiment_text'])
				connection.commit()
			 
		 
		
			connection.close()
			# tf.keras.backend.clear_session()
			# return "Saved successfully."
		    
			return render_template("result.html",result = rows)

		else:
			return "error"			
	      	# print(key)
	      	# print(":")
	      	# print(value)
		

	      

		      
		  
	     

      



      


   

if __name__ == '__main__':
   app.run(debug = True)