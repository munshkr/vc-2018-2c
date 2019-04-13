import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling2D, Input, BatchNormalization
from tensorflow.keras.layers import Convolution2D as Conv2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
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
from tensorflow.keras.models import model_from_json
from pdb import set_trace as st

RHO = 40
DATA_DIR = os.path.join('')


""" session """
from tensorflow.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.compat.v1.Session(config= config) 

set_session(sess)  # set this TensorFlow session as the default session for Keras



def preprocess_labels(y):
    # Rescale
    y = y / RHO
    return y


def preprocess_features(x):
    # Rescale and sample-wise centering
    x = (x - 127.5) / 127.5
    return x

def resize_images(x, w, h):
    img1 = cv2.resize(x[:,:,0], (w, h))
    img2 = cv2.resize(x[:,:,1], (w, h))    
    return np.dstack([img1, img2])

def postprocess_labels(y):
    return np.round(y * RHO).astype(np.int)

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def predict_homographynet(model, X, batch_size=64):
    h, w = model.input_shape[1:3]
    y_pred = []
    for batch in tqdm(list(grouper(X, batch_size))):
        preproc_batch = np.array([resize_images(x, w, h) for x in batch if x is not None])
        preproc_batch = preprocess_features(preproc_batch)
        y_batch = model.predict(preproc_batch)
        y_pred.extend(y_batch)
    return postprocess_labels(np.array(y_pred))

def euclidean_l2_loss(y_true, y_pred):
	diff = K.reshape(y_true, (-1,4,2)) - K.reshape(y_pred, (-1,4,2))
	return K.sqrt(K.sum(K.square(diff), axis=2))


def mace(y_true, y_pred):
	"""Mean Average Corner Error metric"""
	diff = K.reshape(y_true * RHO, (-1,4,2)) - K.reshape(y_pred * RHO, (-1,4,2))
	return K.mean(K.sqrt(K.sum(K.square(diff), axis=2)))  

def build_data_generator(path, batch_size):
	files = glob(os.path.join(path, '*.npz'))
	random.shuffle(files)
	
	for npz in files:
		print(" ")
		print("Loading {}".format(npz))
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

def evaluate_model(model,dirname):
	# we load the patches saved as npz
	evaluation_dir = os.path.join(DATA_DIR, dirname)

	test_generator = build_data_generator(evaluation_dir,64)
	
	scores = model.evaluate(test_generator,steps = 146*5, use_multiprocessing=False, verbose=1)

	for i in range(len(model.metrics_names)):
		print(model.metrics_names[i], scores[i])


def load_and_evaluate(model_name,directory):
	model = load_model(model_name)

	model.compile(optimizer=SGD(lr=0.005, momentum=0.9),
				  loss=euclidean_l2_loss,
				  metrics=['mse', mace])

	print("Evaluating...")
	evaluate_model(model,directory)
	print("Finished evaluation")

def load_model(modelNameInput):
	# load json and create model
	json_file = open(modelNameInput+'.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(modelNameInput+".h5")
	print("Loaded {} from disk".format(modelNameInput))
	return loaded_model

	
if __name__ == "__main__":
	model_name = "train_exp14"

	#test_gen()
	#test_dice(ds_utrecht_smaller)
	#train_and_save(model_name,dataset.ds_utrecht_held_out_augmentated_no_flip_no_discard)
	load_and_evaluate(model_name,"hold_out_rho_40/",)
	#load_and_evaluate(model_name,dataset.ds_singapore_smaller,"val")
	
	#load_and_predict(model_name, ds_utrecht)   
	