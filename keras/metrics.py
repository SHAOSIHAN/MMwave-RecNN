#  --------------- Some loss function ---------------------
import tensorflow as tf
from numpy import asarray
from tensorflow.keras.layers import Flatten

def ssim_loss(y_true, y_pred):
    coef = 0.8
    ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    L2 = tf.losses.mean_squared_error(y_true, y_pred)
    return coef*L2 + (1-coef)*ssim


def PSNR(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0, name=None)


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = Flatten()(y_true)
    y_pred_f = Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def CE_loss(y_true, y_pred):
    '''
    binary cross entropy loss function
    '''
    bce_loss = tf.keras.losses.BinaryCrossentropy()
    loss = bce_loss(y_true, y_pred)
    return loss


def DiceCE_loss(y_true, y_pred):
    coeff = 0.7
    dice_loss = dice_coef_loss(y_true, y_pred)
    ce_loss = CE_loss(y_true, y_pred)
    total_loss = coeff * dice_loss + (1 - coeff) * ce_loss
    return total_loss