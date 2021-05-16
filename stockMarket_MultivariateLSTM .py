# Multivariate single-step vector-output stacked lstm example

import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)-1):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix:out_end_ix+1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# reading data frame ==================================================
df = pd.read_csv('goldETF.csv')

in_cols = ['Open', 'Low', 'Close']
out_cols = ['Close', 'Low']

# choose a number of time steps
n_steps_in, n_steps_out = 45, 1
n_input_nodes = n_steps_in * len(in_cols)
n_hidden_nodes = int((n_input_nodes + n_steps_out)/2)
n_epochs = 100

# Preparing Model for 'Close'=======================================================
j=0
dataset_close = np.empty((df[out_cols[j]].values.shape[0],0))
for i in range(len(in_cols)):
    dataset_close = np.append(dataset_close, df[in_cols[i]].values.reshape(df[in_cols[i]].values.shape[0],1), axis=1)

dataset_close = np.append(dataset_close, df[out_cols[j]].values.reshape(df[out_cols[j]].values.shape[0],1), axis=1)

#==============================================================================
# Preparing Model for 'Low'=======================================================
j=1
dataset_low = np.empty((df[out_cols[j]].values.shape[0],0))
for i in range(len(in_cols)):
    dataset_low = np.append(dataset_low, df[in_cols[i]].values.reshape(df[in_cols[i]].values.shape[0],1), axis=1)

dataset_low = np.append(dataset_low, df[out_cols[j]].values.reshape(df[out_cols[j]].values.shape[0],1), axis=1)

# Scaling dataset
scaler_low = MinMaxScaler(feature_range=(0, 1))
scaler_fit_low = scaler_low.fit(dataset_low)
scaled_data_low = scaler_fit_low.transform(dataset_low)
    
# convert into input/output
X_low, y_low = split_sequences(scaled_data_low, n_steps_in, n_steps_out)

# the dataset knows the number of features, e.g. 2
n_features_low = X_low.shape[2]

# define model ========================================================
model_low = Sequential()
model_low.add(LSTM(n_input_nodes, activation='tanh', return_sequences=True, input_shape=(n_steps_in, n_features_low)))
model_low.add(LSTM(n_hidden_nodes, activation='tanh', return_sequences=True))
model_low.add(LSTM(n_hidden_nodes, activation='tanh'))
model_low.add(Dense(y_low.shape[1]))
model_low.compile(optimizer='adam', loss='mse')

# fit model =======================================================
model_low.fit(X_low, y_low, epochs=n_epochs, batch_size=16, verbose=2)

# demonstrate prediction for past ======================================
n_actualSet, toEnd = 10, 50
fromStart = n_actualSet + toEnd

# For close ===============================================================
if toEnd>0: actualSet_close = dataset_close[-fromStart:-toEnd, -1]
if toEnd==0: actualSet_close = dataset_close[-fromStart:, -1]

# For low ================================================================
if toEnd>0: actualSet_low = dataset_low[-fromStart:-toEnd, -1]
if toEnd==0: actualSet_low = dataset_low[-fromStart:, -1]
yhat_list_low = []
for i in range(n_actualSet, -1, -1):
    if -i-toEnd==0:
        x_input_low = scaled_data_low[-n_steps_in-i-toEnd: , :-1]
    else:
        x_input_low = scaled_data_low[-n_steps_in-i-toEnd: -i-toEnd, :-1]
    
    x_input_low = x_input_low.reshape((1, n_steps_in, n_features_low))    
    yhat_scaled_low = model_low.predict(x_input_low)
    yhat_array_low = np.zeros((n_steps_out, dataset_low.shape[1]))

    for k in range(yhat_scaled_low.shape[1]):
        yhat_array_low[k][-1] = yhat_scaled_low[0][k]
    
    yhat_low = scaler_fit_low.inverse_transform(yhat_array_low)
    yhat_list_low = np.append(yhat_list_low, yhat_low[0][-1])

plt.plot(actualSet_close, label='Actual Close')
plt.plot(yhat_list_low, label='Predicted Low')
plt.plot(actualSet_low, label='Actual Low')
plt.legend()
plt.show()

print(np.around(yhat_list_low))
