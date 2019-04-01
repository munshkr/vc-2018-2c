#!/usr/bin/env python3
import os
import tensorflow.keras.backend as K
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input


WIDTH, HEIGHT = 320, 240
RHO = 32
PATCH_SIZE = 128
# FIXME
DATA_DIR = os.path.join('')


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
                xs_batch = preprocess_input(xs_batch)
                ys_batch = preprocess_labels(ys_batch)

                yield xs_batch, ys_batch


def euclidean_l2_loss(y_true, y_pred):
    diff = K.reshape(y_true, (-1, 4, 2)) - K.reshape(y_pred, (-1, 4, 2))
    return K.sqrt(K.sum(K.square(diff), axis=2))


def mace(y_true, y_pred):
    """Mean Average Corner Error metric"""
    diff = K.reshape(y_true * RHO,
                     (-1, 4, 2)) - K.reshape(y_pred * RHO, (-1, 4, 2))
    return K.mean(K.sqrt(K.sum(K.square(diff), axis=2)))


def homography_regression_model():
    input_shape = (128, 128, 6)

    mobile_net = MobileNetV2(input_shape=input_shape, \
        alpha=1.0, pooling='avg', include_top=False, \
        backend=tf.keras.backend, layers=tf.keras.layers, models=tf.keras.models, utils=tf.keras.utils)

    ### add dense layer + dropout
    x = Flatten()(mobile_net.output)
    x = Dense(512, name='fc1', activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(8, name='fc2', activation=None)(x)

    model = Model(inputs=mobile_net.input, outputs=outputs)

    model.compile(
        optimizer=Adam(lr=0.001),
        loss=euclidean_l2_loss,
        metrics=['mse', mace])

    return model


if __name__ == "__main__":
    import sys

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    model = homography_regression_model()

    print(model.summary())

    print("------------- Training -------------")
    train(model, initial_epoch=0)
