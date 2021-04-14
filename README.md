# StockMarket-Prediction-LSTM

This code gives prediction on stock market datasets using LSTM.

It is a 'one day' prediction on 'Gold ETF' asset data using Multivariate LSTM.

It used all attribute available in the dataset.

Note that all attributes need to be scaled at the same time.

If the activation function is not 'tanh' it cannot be run on GPU.

Play with 'n_actualSet' number to plot data for larger timespans.

Change 'n_steps_out' to get prediction for larger timespans.
