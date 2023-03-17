import tensorflow as tf
from numpy import asarray
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose, ReLU
from tensorflow.keras.layers import Conv2D, Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPool2D, MaxPool1D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Add
from keras.constraints import non_neg
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.activations import sigmoid

## ------------------------ Basic transformation network from 1D spectrum data to 2D image---------------------------
"""
Use Multilayer Perceptron in MLP class and Conv2DTranspose in TransformSD class

"""
class MLP_old(tf.keras.Model):
    # obtain features in the spectrum latent space.
    def __init__(self):
        super().__init__(self)
        # Multilayer Perceptron for real part
        self.rdense = Dense(units=64, activation='sigmoid')
        # Multilayer perceptron for imaginary part
        self.idense = Dense(units=64, activation='sigmoid')
        
        # Multilayer perceptron for feature fusion
        self.fdense = Dense(units=64, activation='sigmoid')
        self.reshape = Reshape((8, 8))
        self.drop = Dropout(0.1)
        
    def call(self, input):
        re, im = tf.split(input, num_or_size_splits=2, axis=1)
        r_out = self.rdense(re)
        
        i_out = self.idense(im)
        
        f_out = tf.concat([r_out, i_out], axis=1)
        f_out = self.drop(f_out)
        out = self.fdense(f_out)
        out = self.reshape(out)
        
        return out

class TransformSD_old(tf.keras.Model):
    # transform the spectrum latent space features to feature maps in the spatial domain
    def __init__(self):
        super().__init__(self)
        self.head_conv = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.dconv1 = Conv2DTranspose(32, (3 ,3), strides = (2,2), padding='same', activation='relu')
        self.dconv2 = Conv2DTranspose(16, (3 ,3), strides = (2,2), padding='same', activation='relu')
        self.dconv3 = Conv2DTranspose(8, (3 ,3), strides = (2,2), padding='same', activation='relu')
        self.dconv4 = Conv2DTranspose(4, (3 ,3), strides = (2,2), padding='same', activation='relu')
        self.dconv5 = Conv2DTranspose(2, (3 ,3), strides = (2,2), padding='same', activation='relu')
        self.end_conv = Conv2D(filters=1, kernel_size=3, padding='same')
    
    def call(self, input):
        input = tf.expand_dims(input, axis=-1)
        out = self.head_conv(input)
        out = self.dconv1(out)
        out = self.dconv2(out)
        out = self.dconv3(out)
        out = self.dconv4(out)
        out = self.dconv5(out)
        out = self.end_conv(out)
        
        return out   


## ------------------------ New transformation network from 1D spectrum data to 2D image---------------------------
"""
Modification:
1. use Residual block instead of Conv2DTranspose to transform the latent 1D spectrum feature to image
(faster convergence, stable training, maybe avoid overfitting) 
2. add a softthresholding function (STF) with learnable threshold as non-linear function in MLP part for
denosing the background noise from the 1D spectrum feature
"""
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
        #self.bn1 = BatchNormalization(axis=3)
        self.conv2 = Conv2D(filters=n_feat, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.relu2 = ReLU()
        #self.bn2 = BatchNormalization(axis=3)
        self.conv3 = Conv2D(filters=n_feat, kernel_size=(1, 1), strides=(1, 1), padding='valid')
        #self.bn3 = BatchNormalization(axis=3)
        self.relu3 = ReLU()
        
        self.short_conv = Conv2D(filters=n_feat, kernel_size=(1, 1), strides=(1, 1), padding='valid')
        #self.bn4 = BatchNormalization(axis=3)
    def call(self, input):
        out = self.conv1(input)
        #out = self.bn1(self.relu1(out))
        out = self.relu1(out)
        out = self.conv2(out)
        #out = self.bn2(self.relu2(out))
        out = self.relu2(out)
        out = self.conv3(out)
        #out = self.bn3(out)
        
        #res = self.bn4(self.short_conv(input))
        res = self.short_conv(input)
        out = self.relu3(res+out)
        return out

class Model_New(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
        self.transformsd = TransformSD()
    
    def call(self, input):
        out = self.mlp(input)
        out = self.transformsd(out)
        return out


## ------------------------ U-shape network --------------------------------
"""
reference: https://arxiv.org/abs/1505.04597
Unet is widely used in image segmention, image restoration.
"""
class down_block(tf.keras.Model):
    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1):
        super().__init__()
        self.conv1 = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")
        self.conv2 = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")
        self.pool = MaxPool2D((2, 2), (2, 2))
    def call(self, input):
        c = self.conv1(input)
        c = self.conv2(c)
        p = self.pool(c)
        return c, p

class up_block(tf.keras.Model):
    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1):
        super().__init__()
        self.us = UpSampling2D((2, 2))
        self.cat = Concatenate()
        self.conv1 = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")
        self.conv2 = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")
    def call(self, x, skip):
        us = self.us(x)
        concat = self.cat([us, skip])
        c = self.conv1(concat)
        c = self.conv2(c)
        return c

class bottleneck(tf.keras.Model):
    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1):
        super().__init__()
        self.conv1 = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")
        self.conv2 = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")
    def call(self, x):
        c = self.conv1(x)
        c = self.conv2(c)
        return c

class UNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        f = [16, 32, 64, 128, 256]
        
        self.down_block1 = down_block(filters=f[0])
        self.down_block2 = down_block(filters=f[1])
        self.down_block3 = down_block(filters=f[2])
        self.down_block4 = down_block(filters=f[3])
        
        self.bn = bottleneck(filters=f[4])
        
        self.up_block1 = up_block(filters=f[3])
        self.up_block2 = up_block(filters=f[2])
        self.up_block3 = up_block(filters=f[1])
        self.up_block4 = up_block(filters=f[0])
        
        self.tail = Conv2D(1, (1, 1), padding="same", activation="sigmoid")
    
    def call(self, x):
        c1, p1 = self.down_block1(x)
        c2, p2 = self.down_block2(p1)
        c3, p3 = self.down_block3(p2)
        c4, p4 = self.down_block4(p3)
        
        bn = self.bn(p4)
        
        u1 = self.up_block1(bn, c4) #8 -> 16
        u2 = self.up_block2(u1, c3) #16 -> 32
        u3 = self.up_block3(u2, c2) #32 -> 64
        u4 = self.up_block4(u3, c1)
        
        out = self.tail(u4)

        return out    

## ------------------------ Residual U-shape network --------------------------------
"""
reference: https://arxiv.org/abs/1904.00592
Use the residual conv block to replace the orginal conv block
make training more stable, fast convergence and better performace
"""
class Res_block(tf.keras.Model):
    '''
    Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, n_feat):
        super().__init__()
        self.conv1 = Conv2D(filters=n_feat, kernel_size=3, padding='same', strides=1, activation='relu')
        self.conv2 = Conv2D(filters=n_feat, kernel_size=3, padding='same', strides=1, activation='relu')
        self.conv3 = Conv2D(filters=n_feat, kernel_size=1, padding="same")
    def call(self, input):
        res = self.conv2(self.conv1(input))
        res += self.conv3(input)
        return res

class down_Resblock(tf.keras.Model):
    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1):
        super().__init__()
        self.conv = Res_block(n_feat=filters)
        self.pool = MaxPool2D((2, 2), (2, 2))
    def call(self, input):
        c = self.conv(input)
        p = self.pool(c)
        return c, p

class up_Resblock(tf.keras.Model):
    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1):
        super().__init__()
        self.us = UpSampling2D((2, 2))
        self.cat = Concatenate()
        self.conv = Res_block(n_feat=filters)
    def call(self, x, skip):
        us = self.us(x)
        concat = self.cat([us, skip])
        c = self.conv(concat)
        return c

class bottleneck(tf.keras.Model):
    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1):
        super().__init__()
        self.conv1 = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")
        self.conv2 = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")
    def call(self, x):
        c = self.conv1(x)
        c = self.conv2(c)
        return c

class ResUNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        f = [16, 32, 64, 128, 256]
        
        self.down_block1 = down_Resblock(filters=f[0])
        self.down_block2 = down_Resblock(filters=f[1])
        self.down_block3 = down_Resblock(filters=f[2])
        self.down_block4 = down_Resblock(filters=f[3])
        
        self.bn = bottleneck(filters=f[4])
        
        self.up_block1 = up_Resblock(filters=f[3])
        self.up_block2 = up_Resblock(filters=f[2])
        self.up_block3 = up_Resblock(filters=f[1])
        self.up_block4 = up_Resblock(filters=f[0])
        
        self.tail = Conv2D(1, (1, 1), padding="same", activation="sigmoid")
    
    def call(self, x):
        c1, p1 = self.down_block1(x)
        c2, p2 = self.down_block2(p1)
        c3, p3 = self.down_block3(p2)
        c4, p4 = self.down_block4(p3)
        
        bn = self.bn(p4)
        
        u1 = self.up_block1(bn, c4) #8 -> 16
        u2 = self.up_block2(u1, c3) #16 -> 32
        u3 = self.up_block3(u2, c2) #32 -> 64
        u4 = self.up_block4(u3, c1)
        
        out = self.tail(u4)

        return out   

## --------------------------Deep unfolding network-------------------------------
"""
Use unfolding stratagy to training network
work for resolution checkboard image dataset but didn't work for A-under-shirt dataset (the loss didn't converge)
"""
class DUN(tf.keras.Model):
    def __init__(self, in_ch=1, out_ch=1, n_feat=32):
        super().__init__()
        # Flexible Gradient Descent Module
        self.phi_1 = Res_block(out_ch=in_ch, n_feat=n_feat)
        self.phi_2 = Res_block(out_ch=in_ch, n_feat=n_feat)
        self.r = tf.Variable(0.1, trainable=True,dtype=tf.float32)
        
        # Informative Proximal Mapping Module
        self.ipmm = ResUNet()#UNet()
    
    def call(self, x):
        # Flexible Gradient Descent Module
        phixsy = self.phi_1(x) - x
        v = x - self.r * self.phi_2(phixsy)
        
        # Informative Proximal Mapping Module
        out = self.ipmm(v)
        return out    
        
class Model_DUN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
        self.transformsd = TransformSD()
        self.basic_block1 = DUN()
    
    def call(self, input):
        out = self.mlp(input)
        out = self.transformsd(out)
        out = self.basic_block1(out)
        return out

if __name__ == '__main__':
    #a = tf.random((1, 402, 1))
    #print(a.shape)
    input_layer = tf.keras.Input(shape=(402), batch_size=4)
    print(input_layer.shape)
    model = Model_New()
    out = model(input_layer)
    print(out.shape)

    
        
        