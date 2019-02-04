from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, MaxPooling2D, Input, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from glob import glob
import numpy as np
import os
import shutil
import random
import cv2
import gc
import matplotlib.pyplot as plt
from itertools import zip_longest
import uuid
from multiprocessing import Pool, cpu_count
from datetime import datetime



WIDTH, HEIGHT = 320, 240
RHO = 32
PATCH_SIZE = 128
# FIXME
DATA_DIR = os.path.join('e:', 'Facultad', 'vc')


def preprocess_features(x):
    # Rescale images
    x = x / 255.0
    # Sample-wise centering
    x -= np.mean(x, axis=0, keepdims=True)
    # Sample-wise std normalization
    x /= (np.std(x, axis=0, keepdims=True) + 1e-7)    
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
                xs_batch = preprocess_features(xs_batch)
                ys_batch = preprocess_labels(ys_batch)

                yield xs_batch, ys_batch

def euclidean_l2_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))

def mace(y_true, y_pred):
    """Mean Average Corner Error metric"""
    return K.mean(RHO * K.sqrt(K.sum(K.square(K.reshape(y_pred, (-1,4,2)) - K.reshape(y_true, (-1,4,2))), \
        axis=-1, keepdims=True)), axis=1)  

def homography_regression_model():
    input_shape = (128, 128, 2)
    filters = 64
    kernel_size = (3, 3)
    conv_strides = (1, 1)
    max_pool_size = (2, 2)
    max_pool_strides = (2, 2)
    
    input_img = Input(shape=input_shape)
     
    x = Conv2D(filters, kernel_size, strides=conv_strides, padding='same', name='conv1', activation='relu')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size, strides=conv_strides, padding='same', name='conv2', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    
    x = Conv2D(filters, kernel_size, strides=conv_strides, padding='same', name='conv3', activation='relu')(x)
    x = BatchNormalization()(x)    
    x = Conv2D(filters, kernel_size, strides=conv_strides, padding='same', name='conv4', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)
   
    x = Conv2D(filters*2, kernel_size, strides=conv_strides, padding='same', name='conv5', activation='relu')(x)
    x = BatchNormalization()(x)    
    x = Conv2D(filters*2, kernel_size, strides=conv_strides, padding='same', name='conv6', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(x)
    
    x = Conv2D(filters*2, kernel_size, strides=conv_strides, padding='same', name='conv7', activation='relu')(x)
    x = BatchNormalization()(x)    
    x = Conv2D(filters*2, kernel_size, strides=conv_strides, padding='same', name='conv8', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)    
    x = Dense(1024, name='fc1', activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(8, name='fc2', activation=None)(x)
    
    model = Model(inputs=input_img, outputs=[out])
    
    model.compile(optimizer=SGD(lr=0.005, momentum=0.9),
                  loss=euclidean_l2_loss,
                  metrics=['mse', mace])

    return model


def train(model):
    batch_size = 16
    total_iterations = 90000
    
    # FIXME
    n_total = len(glob(os.path.join(DATA_DIR, 'test2017', '*.jpg')))
    n_val = round(n_total * 0.05)
    n_train = n_total - n_val

    steps_per_epoch = n_train // batch_size
    epochs = total_iterations // steps_per_epoch

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='min', verbose=1)
    checkpoint = ModelCheckpoint('checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
    
    tb_log_dir = os.path.join("logs", "{}".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    os.makedirs(tb_log_dir, exist_ok=True)
    tensorboard = TensorBoard(log_dir=tb_log_dir, write_images=True)

    train_dir = os.path.join(DATA_DIR, 'dataset', 'train')
    val_dir = os.path.join(DATA_DIR, 'dataset', 'val')
    train_generator = build_data_generator(train_dir, batch_size)
    val_generator = build_data_generator(val_dir, batch_size)

    model.fit_generator(train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=(n_val // batch_size),
        callbacks = [reduce_lr, checkpoint, tensorboard])


if __name__ == "__main__":
    model = homography_regression_model()
    train(model)