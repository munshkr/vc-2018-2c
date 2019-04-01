#!/usr/bin/env python3
import gc
import os
import random
import shutil
import uuid
from datetime import datetime
from glob import glob
from itertools import zip_longest
from multiprocessing import Pool, cpu_count

import numpy as np

import cv2
import keras.backend as K
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from keras.callbacks import (LearningRateScheduler, ModelCheckpoint,
                             ReduceLROnPlateau, TensorBoard)
from keras.layers import (BatchNormalization, Dense, Dropout, Flatten, Input,
                          MaxPooling2D)
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator

from build_dataset import process_image

WIDTH, HEIGHT = 320, 240
RHO = 32
PATCH_SIZE = 128
DATA_DIR = os.path.join('')



def preprocess_features(x):
    # Recale and centering
    x = (x - 127.5) / 127.5
    return x

def preprocess_labels(y):
    # Rescale
    y = y / RHO
    return y

def build_data_generator(path, augment=False, batch_size=64):
    files = glob(os.path.join(path, '*.jpg'))
    if not files:
        return

    if augment:
        # TODO Add more options
        seq = iaa.Sequential([
            #iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
            iaa.Fliplr(0.5), # horizontally flip 50% of the images
            iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
        ])

    xs, ys = [], []
    while True:
        random.shuffle(files)

        for img_path in files:
            img = plt.imread(img_path)

            if augment:
                img = seq.augment_images([img])[0]

            try:
                x, y = process_image(img)
            except:
                print("Image broken? {}".format(img_path))
                raise
            xs.append(x)
            ys.append(y)

            # Yield minibatches
            if len(xs) == batch_size:
                xs = np.array(xs, dtype=np.uint8)
                ys = np.array(ys, dtype=np.int8)

                # Preprocess features and labels
                xs = preprocess_features(xs)
                ys = preprocess_labels(ys)

                yield xs, ys

                xs, ys = [], []

def euclidean_l2_loss(y_true, y_pred):
    diff = K.reshape(y_true, (-1,4,2)) - K.reshape(y_pred, (-1,4,2))
    return K.sqrt(K.sum(K.square(diff), axis=2))

def mace(y_true, y_pred):
    """Mean Average Corner Error metric"""
    diff = K.reshape(y_true * RHO, (-1,4,2)) - K.reshape(y_pred * RHO, (-1,4,2))
    return K.mean(K.sqrt(K.sum(K.square(diff), axis=2)))

def homography_regression_model():
    input_shape = (128, 128, 2)
    filters = 64
    kernel_size = (3, 3)
    conv_strides = (1, 1)

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


def train(model, initial_epoch=None):
    batch_size = 64

    iterations_per_stage = 50000
    stages = 4
    total_iterations = iterations_per_stage * stages

    n_train = len(glob(os.path.join(DATA_DIR, 'dataset', 'train', '*.jpg')))
    n_val = len(glob(os.path.join(DATA_DIR, 'dataset', 'test', '*.jpg')))
    print("Split:", n_train, n_val)

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
    train_dir = os.path.join(DATA_DIR, 'dataset', 'train')
    val_dir = os.path.join(DATA_DIR, 'dataset', 'val')
    train_generator = build_data_generator(train_dir, batch_size=batch_size, augment=True)
    val_generator = build_data_generator(val_dir, batch_size)

    # Fit!
    model.fit_generator(train_generator,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=(n_val // batch_size),
        callbacks = [scheduler, checkpoint, tensorboard])


if __name__ == "__main__":
    import sys

    model = homography_regression_model()

    # For resuming a training job...
    #if os.path.exists('checkpoint.h5'):
    #    model.load_weights('checkpoint.h5')

    initial_epoch = 1
    if len(sys.argv) >= 2:
        initial_epoch = int(sys.argv[1])
        print("Resuming from batch:", initial_epoch)

    train(model, initial_epoch=initial_epoch - 1)