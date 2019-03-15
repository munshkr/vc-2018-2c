import gc
import os
import random
import shutil
import uuid
from glob import glob
from itertools import zip_longest

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from PIL import Image

WIDTH, HEIGHT = 320, 240
RHO = 32
PATCH_SIZE = 128

NUM_SAMPLES_PER_IMAGE = 10
NUM_SAMPLES_PER_ARCHIVE = 1000

NUM_IMAGES_PER_ARCHIVE = NUM_SAMPLES_PER_ARCHIVE // NUM_SAMPLES_PER_IMAGE


def train_test_split(X, test_size=0.2):
    n_test = round(len(X) * test_size)
    random.shuffle(X)
    test_samples, train_samples = X[:n_test], X[n_test:]
    return train_samples, test_samples


def random_patch_positions():
    """Calculate random position for patch P"""
    px = random.randint(RHO, WIDTH - PATCH_SIZE - RHO)
    py = random.randint(RHO, HEIGHT - PATCH_SIZE - RHO)
    return px, py  

def random_delta():
    """Calculate random delta for patch corner distortion"""
    return np.random.randint(-RHO, RHO, size=2)

def extract_from_patch(img, patch):
    """Extract patch from image"""
    a, _, c, _ = patch
    return img[a[1]:c[1], a[0]:c[0],0:3]

def build_random_patch():
    px, py = random_patch_positions()
    return [(px, py),
            (px + PATCH_SIZE, py),
            (px + PATCH_SIZE, py + PATCH_SIZE),
            (px, py + PATCH_SIZE)]

def distort_patch(patch):
    return [p + random_delta() for p in patch]


def get_homography(img, patch, new_patch):
    # Get homography matrix
    H = cv2.getPerspectiveTransform(
        np.float32(patch),
        np.float32(new_patch))

    return H

def transform_image(img,H):
    # Transform image with H
    return cv2.warpPerspective(img, inv(H),
        (img.shape[1], img.shape[0]))  

def process_image(img, debug=False):
    # Resize and convert to grayscale
    resized_img = cv2.resize(img, (WIDTH, HEIGHT))

    if len(resized_img.shape) == 3:
        # si era rgb, convertimos a gris
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    else:
        # si era gris, convertimos a rgb
        gray_img = resized_img
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
    

    assert(len(gray_img.shape) == 2)
    assert(len(resized_img.shape) == 3)
 
    # Build patch, and perturb corners
    patch = build_random_patch()
    distorted_patch = distort_patch(patch)

    # We obtain H from the gray images
    H = get_homography(gray_img, patch, distorted_patch)

    # We apply to the RGB image
    warped_img = transform_image(resized_img, H)
   
    # Extract patches
    a_img = extract_from_patch(resized_img, patch)
    b_img = extract_from_patch(warped_img, patch)
    
      # Stack patch images together
    x = np.dstack([a_img, b_img])
    
    # x should be a 6-channel image
    assert(x.shape[2] == 6)

    # Subtract patches to get delta
    # int8 is good enough because RHO = 32, so, values are in [-32, 32] range
    y = np.subtract(patch, distorted_patch).ravel().astype(np.int8)

    if debug:
        # On debug mode, return also original patch and warped img for plotting
        return x, y, (patch, distorted_patch), (gray_img, warped_img)
    
    return x, y


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def generate_dataset_archives(dirname, X_train, X_val):
    if os.path.exists(dirname):
        print('{} already exists! Delete it if you want to regenerate archives.'.format(dirname))
        return
    
    datasets = dict(train=X_train, val=X_val)

    
    for key, X_set in datasets.items():
        random.shuffle(X_set)
        
        for img_group in grouper(X_set, NUM_IMAGES_PER_ARCHIVE):
            xs, ys = [], []

            cant = 0
            for img_path in img_group: #['test2017/000000368552.jpg']:
                if img_path:
                    for i in range(NUM_SAMPLES_PER_IMAGE):
                        img = plt.imread(img_path)
                        if (cant % 100 == 0):
                            print ("Processing image N° " + str(cant))
                        cant = cant + 1
                        try:
                            x, y = process_image(img)
                            xs.append(x)
                            ys.append(y)
                        except:
                            print("Image broken? {}".format(img_path))

            xs = np.array(xs, dtype=np.uint8)
            ys = np.array(ys, dtype=np.int8)
            
            outfile = os.path.join(dirname, key, '{}.npz'.format(uuid.uuid4()))
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            np.savez_compressed(outfile, x=xs, y=ys)
            print(outfile)


DATA_DIR = os.path.join('')
print(DATA_DIR)
X = glob(os.path.join(DATA_DIR, 'test2017', '*.jpg'))
print("Total", len(X))

# Take a sample of 30000 images (we don't need too many images when doing transfer learning)
np.random.shuffle(X)
X = X[:10000]

X_train, X_val = train_test_split(X, test_size=0.2)
print(len(X_train), len(X_val))

generate_dataset_archives(os.path.join(DATA_DIR, 'datasetColor'), X_train, X_val)