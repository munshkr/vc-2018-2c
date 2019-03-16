import gc
import os
import random
import shutil
import uuid
from datetime import datetime
from glob import glob
from itertools import zip_longest
from multiprocessing import Pool, cpu_count

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.backend import set_session
from tensorflow.keras.callbacks import (LearningRateScheduler, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Convolution2D as Conv2D
from tensorflow.keras.layers import (Dense, Dropout, Flatten, Input, Lambda,
                                     MaxPooling2D, concatenate)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adagrad, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import mobilenet_v2

WIDTH, HEIGHT = 320, 240
RHO = 32
PATCH_SIZE = 128
# FIXME
DATA_DIR = os.path.join('')


""" session """
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.compat.v1.Session(config= config) 

set_session(sess)  # set this TensorFlow session as the default session for Keras




def preprocess_features(x):
    # Rescale images
    #x = x / 255.0
    # Sample-wise centering
    #x -= np.mean(x, axis=0, keepdims=True)
    # Sample-wise std normalization
    #x /= (np.std(x, axis=0, keepdims=True) + 1e-7)    
    x = (x - 127.5) / 127.5
    return x

def preprocess_labels(y):
    # Rescale
    y = y / RHO
    return y

def build_data_generator(path, batch_size=64):
    while True:
        files = glob(os.path.join(path, '*.npz'))
        random.shuffle(files)
        
        for npz in files:
            archive = np.load(npz)
            xs = archive['x']
            ys = archive['y']

            # Yield minibatches
            for i in range(0, len(xs), batch_size):
                end_i = min(i + batch_size, len(xs))
                xs_batch = xs[i:end_i]
                ys_batch = ys[i:end_i]
                
                # Preprocess features and labels
                xs_batch = mobilenet_v2.preprocess_input(xs_batch,backend=K)

                ys_batch = preprocess_labels(ys_batch)

                yield [xs_batch[:,:,:,0:3], xs_batch[:,:,:,3:6]], ys_batch

def euclidean_l2_loss(y_true, y_pred):
    diff = K.reshape(y_true, (-1,4,2)) - K.reshape(y_pred, (-1,4,2))
    return K.sqrt(K.sum(K.square(diff), axis=2))

def mace(y_true, y_pred):
    """Mean Average Corner Error metric"""
    diff = K.reshape(y_true * RHO, (-1,4,2)) - K.reshape(y_pred * RHO, (-1,4,2))
    return K.mean(K.sqrt(K.sum(K.square(diff), axis=2)))  


def train(model, initial_epoch=None):
    batch_size = 16

    iterations_per_stage = 50000
    stages = 4
    total_iterations = iterations_per_stage * stages
    
    # FIXME
    n_total = len(glob(os.path.join(DATA_DIR, 'test2017', '*.jpg')))
    n_val = round(n_total * 0.2)
    n_train = n_total - n_val
    print(n_total)
    print(n_val)


    steps_per_epoch = n_train // batch_size
    epochs = total_iterations // steps_per_epoch
    epochs_per_stage = iterations_per_stage // steps_per_epoch

    print("epochs:", epochs)
    print("epochs per stage:", epochs_per_stage)

    # Callbacks
    def scheduler_fn(epoch, lr):
        if epoch > 0 and epoch % epochs_per_stage == 0:
            return lr * 0.1
        else:
            return lr
    scheduler = LearningRateScheduler(scheduler_fn, verbose=1)
    checkpoint = ModelCheckpoint('checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
    
    tb_log_dir = os.path.join("logs", "{}".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    os.makedirs(tb_log_dir, exist_ok=True)
    tensorboard = TensorBoard(log_dir=tb_log_dir, write_images=True)

    # Build data generators
    train_dir = os.path.join(DATA_DIR, 'datasetColor', 'train')
    val_dir = os.path.join(DATA_DIR, 'datasetColor', 'val')
    train_generator = build_data_generator(train_dir, batch_size)
    val_generator = build_data_generator(val_dir, batch_size)

    # Fit!
    model.fit_generator(train_generator,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=(n_val // batch_size),
        callbacks = [scheduler, checkpoint, tensorboard])


def homography_regression_model_transfer_learning():
    input_shape = (128, 128, 3)
    filters = 64
    kernel_size = (3, 3)
    conv_strides = (1, 1)
    

    ### ---------- mobilenet 1
    input_img1 = Input(shape=input_shape)
    mobileNet1 = mobilenet_v2.MobileNetV2(input_shape=input_shape, \
                    alpha=1.0, include_top=False, weights="imagenet",layerNameSuffix="_1",\
                    backend = tf.keras.backend, layers = tf.keras.layers, models = tf.keras.models, utils = tf.keras.utils)
    
    out1 = Flatten()(mobileNet1.output)
    inp1 = mobileNet1.input
    modelMobileNet1 = Model(inp1, out1, name="mobileNet1")
    for layer in modelMobileNet1.layers:
        layer.trainable = False


    ### ---------- mobilenet 2
    input_img2 = Input(shape=input_shape)
    mobileNet2 = mobilenet_v2.MobileNetV2(input_shape=input_shape, \
                    alpha=1.0, include_top=False, weights="imagenet",layerNameSuffix="_2",\
                    backend = tf.keras.backend, layers = tf.keras.layers, models = tf.keras.models, utils = tf.keras.utils)
    out2 = Flatten()(mobileNet2.output)
    inp2 = mobileNet2.input
    modelMobileNet2 = Model(inp2, out2, name="mobileNet2")
    for layer in modelMobileNet2.layers:
        layer.trainable = False


    ### merge
    merged = concatenate([out1, out2])
    

    ### we add dense layers + dropouts
    x = Dense(1024, name='fc1', activation='relu')(merged)
    x = Dropout(0.5)(x)    
    x = Dense(1024, name='fc2', activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(8, name='fc3', activation=None)(x)

    model = Model(inputs=[modelMobileNet1.input, modelMobileNet2.input], outputs=[out])
    
    model.compile(optimizer=SGD(lr=0.005, momentum=0.9),
                  loss=euclidean_l2_loss,
                  metrics=['mse', mace])

    return model

if __name__ == "__main__":
    import sys

    model = homography_regression_model_transfer_learning()

    #print(model.summary())

    #print("------------- Training -------------")
    train(model, initial_epoch=0)
