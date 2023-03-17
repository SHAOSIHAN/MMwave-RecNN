#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  

import tensorflow as tf
from numpy import asarray
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from scipy.io import loadmat
import numpy as np

def ssim_loss(y_true, y_pred):
    ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    L2 = tf.losses.mean_squared_error(y_true, y_pred)
    return L2 + ssim

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = Flatten()(y_true)
    y_pred_f = Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def PSNR(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0, name=None)



def main(args):

	data = loadmat(r'data/activations_AaltoLog202302142234.mat')

	inputsTrain = np.array(data['inputsTrain'])
	labelsTrain = np.array(data['labelsTrain']).reshape(-1, 256, 256, 1)
	labelsTrain = labelsTrain/255.0
	inputsPred = np.array(data['inputsPred'])
	labelsPred = np.array(data['labelsPred'])
	labelsPred = labelsPred/255.0
	np.savetxt('labels20220801.txt', labelsPred, delimiter = " ")
	labelsPred = labelsPred.reshape(-1, 256, 256, 1)
	
	
	numInputs = 802
		
	#train_dataset = tf.data.Dataset.from_tensor_slices((inputsTrain, labelsTrain))
 
	# build a tensorboard class for visualizing some parameters
	tbCallBack = tf.keras.callbacks.TensorBoard(log_dir="./logs", write_images=True)
	mcp_save = ModelCheckpoint('best_model.hdf5', save_best_only=True, monitor='val_loss', mode='min')

	model = Sequential()
	model.add(Dense(100*2*2, input_shape=(numInputs,), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Reshape((2,2,100)))
	model.add(Conv2DTranspose(50,(3,3), strides = (2,2), padding='same',activation='relu'))
	model.add(Conv2DTranspose(25,(3,3), strides = (2,2), padding='same',activation='relu'))
	model.add(Conv2DTranspose(12,(3,3), strides = (2,2), padding='same',activation='relu'))
	model.add(Conv2DTranspose(6,(3,3), strides = (2,2), padding='same',activation='relu'))
	model.add(Conv2DTranspose(3,(3,3), strides = (2,2), padding='same',activation='relu'))
	model.add(Conv2DTranspose(2,(3,3), strides = (2,2), padding='same',activation='relu'))
	model.add(Conv2DTranspose(1,(3,3), strides = (2,2), padding='same',activation='sigmoid'))
	#model.add(Reshape((256*256,1)))
	#model.add(Flatten())
	model.summary()
	# exit()
	
	# Changed the learning rate on 19th February 2021 for improved convergence of *both* training and validating sets
	opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
	
	# model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae']) # Changed back to MSE as the sigmoid at output makes it work again, 23rd February 2021, ATa
	
	model.compile(optimizer=opt, loss=dice_coef_loss, metrics=['mae']) # Changed back to MSE as the sigmoid at output makes it work again, 23rd February 2021, ATa
	
	history = model.fit(inputsTrain, labelsTrain, steps_per_epoch = 500, epochs = 500, validation_data=(inputsPred, labelsPred), validation_steps = 1, callbacks=[tbCallBack, mcp_save])
	
	# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mae'])
	# history = model.fit(inputs, labels, steps_per_epoch = 1000, epochs=200, validation_data=(inputs, labels), validation_steps = 3)

	prediction = model.predict(inputsPred).reshape(-1, 256*256*1)
	
	np.savetxt('prediction.txt', prediction, delimiter=" ")
	np.savetxt('loss.txt', history.history['loss'])
	np.savetxt('valLoss.txt', history.history['val_loss'])
	np.savetxt('mae.txt', history.history['mae'])
	np.savetxt('valMae.txt', history.history['val_mae'])
	
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights("base_model.h5")
	print("Saved model to disk!")
	
	return 0
	
if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
