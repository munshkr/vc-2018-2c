#!/usr/bin/env python3
import os
import shutil
from glob import glob
import random
from sklearn.model_selection import train_test_split

DATA_DIR = '.'
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')

def train_test_split(X, test_size=0.2):
    n_test = round(len(X) * test_size)
    random.shuffle(X)
    test_samples, train_samples = X[:n_test], X[n_test:]
    return train_samples, test_samples


X = glob(os.path.join(DATA_DIR, 'test2017', '*.jpg'))
X_train, X_val = train_test_split(X, test_size=0.2)

for ds, name in zip([X_train, X_val], ['train', 'test']):
    for x in ds:
        basename = os.path.basename(x)
        dst = os.path.join(DATASET_DIR, name, basename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(x, dst)
