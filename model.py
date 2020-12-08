import os
import math
import numpy as np
import datetime as dt

from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

class Model():
	def __init__(self):
		self.model = Sequential()

	def load_model(self, filepath):
		self.model = load_model(filepath)

	def build_model(self, configs):

		for layer in configs['model']['layers']:

			if 'neurons' in layer:
				neurons = layer['neurons']
			else:
				neurons = None

			if 'rate' in layer:
				dropout_rate = layer['rate']
			else:
				dropout_rate = None

			if 'activation' in layer:
				activation = layer['activation']
			else:
				activation = None

			if 'return_seq' in layer:
				return_seq = layer['return_seq']
			else:
				return_sql = None

			if 'input_timesteps' in layer:
				input_timesteps = layer['input_timesteps']
			else:
				input_timesteps = None

			if 'input_dim' in layer:
				input_dim = layer['input_dim']
			else:
				input_dim = None

			if layer['type'] == 'dense':
				self.model.add(
					Dense(
						neurons,
						activation=activation
					)
				)

			if layer['type'] == 'lstm':
				self.model.add(
					LSTM(
						neurons,
						input_shape=(
							input_timesteps,
							input_dim
						),
						return_sequences=return_seq
					)
				)

			if layer['type'] == 'dropout':
				self.model.add(
					Dropout(
						dropout_rate
					)
				)

		self.model.compile(
			loss=configs['model']['loss'], optimizer=configs['model']['optimizer']
		)

	def train(self, x, y, epochs, batch_size, save_dir):
		save_fname =\
		os.path.join(
			save_dir, '%s-e%s.h5' % (
				dt.datetime.now().strftime('%d%m%Y-%H%M%S'),
				str(epochs)
			)
		)
		callbacks = [
			EarlyStopping(
				monitor='val_loss',
				patience=2
			),
			ModelCheckpoint(
				filepath=save_fname, monitor='val_loss', save_best_only=True
			)
		]
		self.model.fit(
			x,
			y,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks
		)
		self.model.save(save_fname)

	def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):

		save_fname = os.path.join(
			save_dir, '%s-e%s.h5' % (
				dt.datetime.now().strftime('%d%m%Y-%H%M%S'),
				str(epochs)
				)
			)
		callbacks = [
			ModelCheckpoint(
				filepath=save_fname,
				monitor='loss',
				save_best_only=True
			)
		]
		self.model.fit_generator(
			data_gen,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			callbacks=callbacks,
			workers=1
		)

	def predict_point_by_point(self, data):
		#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
		predicted = self.model.predict(data)
		predicted = np.reshape(predicted, (predicted.size,))
		return predicted

	def predict_sequences_multiple(self, data, window_size, prediction_len):
		#Predict sequence of 50 steps before shifting prediction run forward by 50 steps
		prediction_seqs = []
		ran = int(len(data)/prediction_len)
		for i in range(ran):
			curr_frame = data[i*prediction_len]
			predicted = []
			for j in range(prediction_len):
				predicted.append(
					self.model.predict(
						curr_frame[newaxis,:,:]
					)[0,0]
				)
				curr_frame = curr_frame[1:]
				curr_frame = np.insert(
					curr_frame,
					[window_size-2],
					predicted[-1],
					axis=0
				)
			prediction_seqs.append(predicted)
		return prediction_seqs

	def predict_sequence_full(self, data, window_size):
		#Shift the window by 1 new prediction each time, re-run predictions on new window
		curr_frame = data[0]
		predicted = []
		for i in range(len(data)):
			predicted.append(
				self.model.predict(
					curr_frame[newaxis,:,:]
				)[0,0]
			)
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(
				curr_frame,
				[window_size-2],
				predicted[-1],
				axis=0
			)
		return predicted
