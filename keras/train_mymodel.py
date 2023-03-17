#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  

import tensorflow as tf
from numpy import asarray
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Conv2D, Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from keras.constraints import non_neg
from scipy.io import loadmat
import numpy as np
import os
import tensorflow_addons as tfa
from metrics import ssim_loss, DiceCE_loss, PSNR
from Models import Model_New

class MLP(tf.keras.Model):
    # obtain features in the spectrum latent space.
    def __init__(self):
        super().__init__(self)
        # Multilayer Perceptron for real part
        #self.rdense = Dense(units=64, activation='sigmoid')
        self.rdense = Dense(units=64)
        # Multilayer perceptron for imaginary part
        #self.idense = Dense(units=64, activation='sigmoid')
        self.idense = Dense(units=64)
        # Multilayer perceptron for feature fusion
        #self.fdense = Dense(units=64, activation='sigmoid')
        self.fdense = Dense(units=64)
        # AdaptiveSoftThreshold Block
        self.thres1 = tf.Variable(0.1, trainable=True, constraint=non_neg(), dtype=tf.float32)
        self.thres2 = tf.Variable(0.1, trainable=True, constraint=non_neg(), dtype=tf.float32)
        self.thres3 = tf.Variable(0.1, trainable=True, constraint=non_neg(), dtype=tf.float32)

        self.reshape = Reshape((8, 8))
        self.drop = Dropout(0.5)
        
    def call(self, input):
        print(input.shape)
        re, im = tf.split(input, num_or_size_splits=2, axis=1)
        r_out = self.rdense(re)
        r_out = tf.multiply(tf.sign(r_out), tf.maximum(tf.abs(r_out) - self.thres1, 0))
        i_out = self.idense(im)
        i_out = tf.multiply(tf.sign(i_out), tf.maximum(tf.abs(i_out) - self.thres2, 0))
        
        f_out = tf.concat([r_out, i_out], axis=1)
        f_out = self.drop(f_out)
        out = self.fdense(f_out)
        out = tf.multiply(tf.sign(out), tf.maximum(tf.abs(out) - self.thres3, 0))
        out = self.reshape(out)
        
        return out

class TransformSD(tf.keras.Model):
    # transform the spectrum latent space features to feature maps in the spatial domain
    def __init__(self):
        super().__init__(self)
        self.head_conv = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.us = UpSampling2D((2, 2))
        self.res1 = Res_block(n_feat=64)
        self.res2 = Res_block(n_feat=32)
        self.res3 = Res_block(n_feat=16)
        self.res4 = Res_block(n_feat=8)
        self.res5 = Res_block(n_feat=4)
        
        self.end_conv = Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')
    
    def call(self, input):
        input = tf.expand_dims(input, axis=-1)
        out = self.head_conv(input)
        out = self.res1(out)
        out = self.us(out)
        out = self.res2(out)
        out = self.us(out)
        out = self.res3(out)
        out = self.us(out)
        out = self.res4(out)
        out = self.us(out)
        out = self.res5(out)
        out = self.us(out)
        out = self.end_conv(out)
        
        return out

class Res_block(tf.keras.Model):
    '''
    Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, n_feat):
        super().__init__()
        self.conv1 = Conv2D(filters=n_feat, kernel_size=(1, 1), strides=(1, 1), padding='valid')
        self.relu1 = ReLU()
        self.bn1 = BatchNormalization(axis=3)
        self.conv2 = Conv2D(filters=n_feat, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.relu2 = ReLU()
        self.bn2 = BatchNormalization(axis=3)
        self.conv3 = Conv2D(filters=n_feat, kernel_size=(1, 1), strides=(1, 1), padding='valid')
        self.bn3 = BatchNormalization(axis=3)
        self.relu3 = ReLU()
        
        self.short_conv = Conv2D(filters=n_feat, kernel_size=(1, 1), strides=(1, 1), padding='valid')
        self.bn4 = BatchNormalization(axis=3)
    def call(self, input):
        out = self.conv1(input)
        out = self.bn1(self.relu1(out))
        out = self.conv2(out)
        out = self.bn2(self.relu2(out))
        out = self.conv3(out)
        out = self.bn3(out)
        
        res = self.bn4(self.short_conv(input))
        out = self.relu3(res+out)
        return out
    

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
        self.transformsd = TransformSD()
    
    def call(self, input):
        out = self.mlp(input)
        out = self.transformsd(out)
        return out

def main(args):

    data = loadmat(r'data/activations_AaltoLog202302142234.mat')

    inputsTrain = np.array(data['inputsTrain'])
    labelsTrain = np.array(data['labelsTrain'], dtype=np.float32).reshape(-1, 256, 256, 1)
    labelsTrain = tfa.image.gaussian_filter2d(labelsTrain, filter_shape=19)/255.0
    inputsPred = np.array(data['inputsPred'])
    labelsPred = np.array(data['labelsPred'], dtype=np.float32).reshape(-1, 256, 256, 1)
    labelsPred = tfa.image.gaussian_filter2d(labelsPred, filter_shape=19)/255.0
    
    np.savetxt('labels20220801.txt', np.array(labelsPred).reshape(-1, 256*256), delimiter = " ")
    
    numInputs = 402
    
    #train_dataset = tf.data.Dataset.from_tensor_slices((inputsTrain, labelsTrain))
 
    # build a tensorboard class for visualizing some parameters
    tbCallBack = TensorBoard(log_dir="./logs", write_images=True)
    #reduce_lr_loss = LearningRateScheduler(scheduler)
    
    model = Model()
    #model.summary()
    # exit()
    
    # Changed the learning rate on 19th February 2021 for improved convergence of *both* training and validating sets
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
    
    # model.compile(optimizer=opt, loss='mean_squared_error' or 'mean_absolute_error', metrics=['mae']) # Changed back to MSE as the sigmoid at output makes it work again, 23rd February 2021, ATa
    
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae']) # Changed back to MSE as the sigmoid at output makes it work again, 23rd February 2021, ATa
    
    history = model.fit(inputsTrain, labelsTrain, batch_size=64, epochs = 300, validation_data=(inputsPred, labelsPred), validation_steps = 2, callbacks=[tbCallBack])
    
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mae'])
    # history = model.fit(inputs, labels, steps_per_epoch = 1000, epochs=200, validation_data=(inputs, labels), validation_steps = 3)

    prediction = model.predict(inputsPred).reshape(-1, 256*256*1)
    
    np.savetxt('Mymodel_Aprediction.txt', prediction, delimiter=" ")
    np.savetxt('loss.txt', history.history['loss'])
    np.savetxt('valLoss.txt', history.history['val_loss'])
    #np.savetxt('PSNR.txt', history.history['PSNR'])
    #np.savetxt('valPSNR.txt', history.history['val_PSNR'])
    
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)  
    model.save_weights("model.h5")
    print("Saved model to disk!")
    
    return 0
    
if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))