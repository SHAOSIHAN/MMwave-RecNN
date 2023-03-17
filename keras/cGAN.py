from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from numpy import asarray
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ReLU, LeakyReLU
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
        self.relu1 = LeakyReLU(0.2)
        self.bn1 = BatchNormalization(axis=3)
        self.conv2 = Conv2D(filters=n_feat, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.relu2 = LeakyReLU(0.2)
        self.bn2 = BatchNormalization(axis=3)
        self.conv3 = Conv2D(filters=n_feat, kernel_size=(1, 1), strides=(1, 1), padding='valid')
        self.bn3 = BatchNormalization(axis=3)
        self.relu3 = LeakyReLU(0.2)
        
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
    
class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
        self.transformsd = TransformSD()
    
    def call(self, input):
        out = self.mlp(input)
        out = self.transformsd(out)
        return out

class Discriminator(tf.keras.Model):
    def __init__(self, padding='same', use_bias=True):
        super().__init__()
        self.conv1 = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding=padding, use_bias=use_bias)
        self.act1 = LeakyReLU(0.2)
        
        self.conv2 = Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding=padding, use_bias=use_bias)
        self.act2 = LeakyReLU(0.2)
        self.drop2 = Dropout(0.3)
        
        self.conv3 = Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding=padding, use_bias=use_bias)
        self.act3 = LeakyReLU(0.2)
        self.drop3 = Dropout(0.3)
        
        self.conv4 = Conv2D(512, kernel_size=(5, 5), strides=(2, 2), padding=padding, use_bias=use_bias)
        self.act4 = LeakyReLU(0.2)
        
        self.flatten = Flatten()
        self.drop4 = Dropout(0.3)
        self.dense = Dense(1)
    def call(self, input):
        x = self.conv1(input)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.drop2(self.act2(x))
        
        x = self.conv3(x)
        x = self.drop3(self.act3(x))
        
        x = self.conv4(x)
        x = self.act4(x)
        
        x = self.flatten(x)
        x = self.drop4(x)
        out = self.dense(x)
        return out
    



batch_size = 64
num_channels = 1
num_classes = 10
image_size = 28
latent_dim = 128

# We'll use all the available examples from both the training and test
# sets.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])

# Scale the pixel values to [0, 1] range, add a channel dimension to
# the images, and one-hot encode the labels.
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
all_labels = keras.utils.to_categorical(all_labels, 10)

# Create tf.data.Dataset.
dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

print(f"Shape of training images: {all_digits.shape}")
print(f"Shape of training labels: {all_labels.shape}")

generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)

