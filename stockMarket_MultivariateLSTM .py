# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 13:34:58 2020

@author: Kianoosh Keshavarzian
"""

# Multivariate single-step vector-output stacked lstm example

import pandas as pd
import numpy as np
from numpy import array
from numpy import hstack
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# reading data frame ==================================================

df = pd.read_csv('goldETF.csv')

# define input sequence
in_seq1 = array(df['High'].values)
in_seq2 = array(df['Low'].values)
in_seq3 = array(df['Open'].values)
in_seq4 = array(df['Close'].values)
in_seq5 = array(df['Volume'].values)

out_seq = array(df['Close'].values)

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
in_seq4 = in_seq4.reshape((len(in_seq4), 1))
in_seq5 = in_seq5.reshape((len(in_seq5), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, in_seq3, in_seq4, in_seq5, out_seq))

# scaling dataset ===============================================
in_seq = np.concatenate((in_seq1, in_seq2, in_seq3, in_seq4, in_seq5, out_seq), axis=0)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_all = scaler.fit(in_seq)
scaled_in_seq1 = scaler_all.transform(in_seq1)
scaled_in_seq2 = scaler_all.transform(in_seq2)
scaled_in_seq3 = scaler_all.transform(in_seq3)
scaled_in_seq4 = scaler_all.transform(in_seq4)
scaled_in_seq5 = scaler_all.transform(in_seq5)
scaled_out_seq = scaler_all.transform(out_seq)

scaled_data = hstack((scaled_in_seq1, scaled_in_seq2, scaled_in_seq3, scaled_in_seq4, scaled_in_seq5, scaled_out_seq))

# choose a number of time steps
n_steps_in, n_steps_out = 100, 1

# convert into input/output
X, y = split_sequences(scaled_data, n_steps_in, n_steps_out)

# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]

# define model ========================================================
model = Sequential()
model.add(LSTM(100, activation='tanh', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='tanh'))
model.add(Dense(y.shape[1]))
model.compile(optimizer='adam', loss='mse')

# fit model =======================================================
model.fit(X, y, epochs=100, batch_size=16, verbose=2)
train_acc = model.evaluate(X, y, batch_size=16, verbose=2)

model.save('mdl.pmdl')

# demonstrate prediction for past ======================================
n_actualSet, toEnd = 50, 0
fromStart = n_actualSet + toEnd
if toEnd>0: actualSet = dataset[-fromStart:-toEnd, -1]
if toEnd==0: actualSet = dataset[-fromStart:, -1]
yhat_npArray = []
for i in range(n_actualSet, 0, -1):
    x_input = scaled_data[-n_steps_in-i-toEnd: -i-toEnd, :-1]
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input)
    yhat = scaler_all.inverse_transform(yhat)
    yhat_npArray = np.append(yhat_npArray, yhat)

#plt.plot(yhat_npArray-mean_difference(yhat_npArray, actualSet), label='Predicted')
plt.plot(yhat_npArray, label='Predicted')
plt.plot(actualSet, label='Actual')
plt.legend()
plt.show()
