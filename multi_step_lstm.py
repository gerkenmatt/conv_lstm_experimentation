from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from math import sqrt
from matplotlib import pyplot
from numpy import array

# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
	raw_values = series.values
	# transform data to be stationary
	diff_series = difference(raw_values, 1)
	diff_values = diff_series.values
	diff_values = diff_values.reshape(len(diff_values), 1)
	# rescale values to -1, 1
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_values = scaler.fit_transform(diff_values)
	scaled_values = scaled_values.reshape(len(scaled_values), 1)
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(scaled_values, n_lag, n_seq)
	supervised_values = supervised.values
	print("SUPERVISED VALUES SHAPE: ", str(supervised_values.shape))
	print("SUPERVISED VALUES: ",  str(supervised_values))

	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return scaler, train, test



# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
	# reshape training into [samples, timesteps, features]
	X, y = train[:, 0:n_lag], train[:, n_lag:]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	# design network
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam')
	# fit network
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
		model.reset_states()
	return model


# transform series into train and test sets for supervised learning
def prepare_raw_data(series, n_test, n_lag, n_seq):
	# extract raw values
	raw_values = series.values
	# transform data to be stationary
	diff_series = difference(raw_values, 1)
	diff_values = diff_series.values
	diff_values = diff_values.reshape(len(diff_values), 1)
	# rescale values to -1, 1
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_values = scaler.fit_transform(diff_values)
	scaled_values = scaled_values.reshape(len(scaled_values), 1)

	# split into train and test sets
	train, test = scaled_values[0:-n_test], scaled_values[-n_test:]
	return scaler, train.flatten(), test.flatten()	

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]

# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
	forecasts = list()
	print("TEST SHAPE: ", str(test.shape))
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		print("X shape: ", str(X.shape))
		print("y shape: ", str(y.shape))
		# make forecast
		forecast = forecast_lstm(model, X, n_batch)
		# store the forecast
		forecasts.append(forecast)
	return forecasts



# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def fit_cnn_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):

	
	# choose a number of time steps
	n_steps = 4

	print("TRAIN SHAPE BEFORE SPLIT: ", str(train.shape))
	# split into samples
	X, y = split_sequence(train, n_steps)

	print("TRAIN SHAPE: ", str(X.shape))
	# X, y = train[:, 0:n_lag], train[:, n_lag:]
	# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
	n_features = 1
	n_seq = 2
	n_steps = 2
	X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
	print("RESHAPED TRAIN SHAPE: ", str(X.shape))

	model = Sequential()
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	model.add(LSTM(50, activation='relu'))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse')

	for i in range(nb_epoch):
		if i % 10 == 0: 
			print("Epoch: ", str(i))
		model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
		model.reset_states()
	return model

# make one forecast with an LSTM,
def forecast_cnn_lstm(model, X, n_batch):

	n_features = 1
	n_seq = 2
	n_steps = 2

	print("   FORECAST X SHAPE: ", str(X.shape))
	# reshape input pattern to [samples, timesteps, features]
	# X = X.reshape(1, 1, len(X))
	X = X.reshape((1, n_seq, n_steps, n_features))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]

# evaluate the persistence model
def make_cnn_forecasts(model, n_batch, train, test, n_lag, n_seq):
	n_seq = 2
	n_steps = 4

	test, y = split_sequence(test, n_steps)
	print("TEST SHAPE: ", str(test.shape))
	forecasts = list()
	for i in range(len(test) -1 ):
		# X, y = test[i, 0:n_lag], test[i, n_lag:]
		X, y = test[i], test[i] #we dont even use y
		print("X shape: ", str(X.shape))
		# print("y shape: ", str(y.shape))
		# make forecast
		forecast = forecast_cnn_lstm(model, X, n_batch)
		# store the forecast
		forecasts.append(forecast)
	return forecasts



# invert differenced forecast
def inverse_difference(last_ob, forecast):
	# invert first forecast
	inverted = list()
	inverted.append(forecast[0] + last_ob)
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted


# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
	inverted = list()
	for i in range(len(forecasts)):
		# create array from forecast
		forecast = array(forecasts[i])
		forecast = forecast.reshape(1, len(forecast))
		# invert scaling
		inv_scale = scaler.inverse_transform(forecast)
		inv_scale = inv_scale[0, :]
		# invert differencing
		index = len(series) - n_test + i - 1
		last_ob = series.values[index]
		inv_diff = inverse_difference(last_ob, inv_scale)
		# store
		inverted.append(inv_diff)
	return inverted

# invert differenced forecast
def inverse_cnn_difference(last_ob, forecast):
	# invert first forecast
	inverted = list()
	inverted.append(forecast[0] + last_ob)

	return inverted

# inverse data transform on forecasts
def inverse_cnn_transform(series, forecasts, scaler, n_test):
	inverted = list()
	for i in range(len(forecasts)):
		# create array from forecast
		forecast = array(forecasts[i])
		forecast = forecast.reshape(1, len(forecast))
		# invert scaling
		inv_scale = scaler.inverse_transform(forecast)
		inv_scale = inv_scale[0, :]
		# invert differencing
		index = len(series) - n_test + i - 1
		last_ob = series.values[index]
		inv_diff = inverse_difference(last_ob, inv_scale)
		# store
		inverted.append(inv_diff)
	return inverted



# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))

# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	pyplot.plot(series.values)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i - 1
		print("OFF_S: ", str(off_s))
		off_e = off_s + len(forecasts[i]) + 1
		print("OFF_E: ", str(off_e))
		xaxis = [x for x in range(off_s, off_e)]
		print("XAXIS: ", str(xaxis))
		yaxis = [series.values[off_s]] + forecasts[i]
		print("YAXIS: ", str(yaxis))
		pyplot.plot(xaxis, yaxis, color='red')
	# show the plot
	pyplot.show()

# plot the forecasts in the context of the original dataset
def plot_cnn_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	pyplot.plot(series.values)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i*3 - 1 # start offset
		off_e = off_s + len(forecasts[i]) + 1 # end offset
		xaxis = [x for x in range(off_s, off_e)]
		yaxis = [series.values[off_s]] + forecasts[i]
		pyplot.plot(xaxis, yaxis, color='red')
	# show the plot
	pyplot.show()

# load dataset
print("LOAD DATASET")
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# configure
n_lag = 1
n_seq = 3
n_test = 10
n_epochs = 20 #1500
n_batch = 1
n_neurons = 1

# prepare data
print("PREPARE DATA")

scaler, train, test = prepare_raw_data(series, n_test, n_lag, n_seq)
scaler2, train2, test2 = prepare_data(series, n_test, n_lag, n_seq)
print("TRAIN2 SHAPE: ", str(train2.shape))

# fit model
print("FIT LSTM")
model =fit_cnn_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons) 
model2 = fit_lstm(train2, n_lag, n_seq, n_batch, n_epochs, n_neurons)

# make forecasts
print("MAKE FORECASTS")
forecasts = make_cnn_forecasts(model, n_batch, train, test, n_lag, n_seq)
forecasts2 = make_forecasts(model2, n_batch, train2, test2, n_lag, n_seq)


# inverse transform forecasts and test
print("INVERSE TRANSFORM")
forecasts2 = inverse_transform(series, forecasts2, scaler2, n_test+2)
actual2 = [row[n_lag:] for row in test2]
actual2 = inverse_transform(series, actual2, scaler2, n_test+2)

forecasts = inverse_transform(series, forecasts, scaler, n_test+2)


# evaluate forecasts
print("EVALUATE FORECASTS")
evaluate_forecasts(actual2, forecasts2, n_lag, n_seq)

# plot forecasts
print("PLOT FORECASTS")
plot_forecasts(series, forecasts2, n_test+2)
plot_cnn_forecasts(series, forecasts, n_test+2)
print("********************DONE")