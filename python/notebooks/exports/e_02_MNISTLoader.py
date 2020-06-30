# module automatically generated from 02_MNISTLoader.ipynb
# to change this code, please edit the appropriate notebook and re-export, rather than editing this script directly

import gzip
import pickle

from fastai import datasets
from torch import tensor


def loadMNIST():
    MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'
    path = datasets.download_data(MNIST_URL, ext='.gz')

    with gzip.open(path, 'rb') as file:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(file,
                                                        encoding='latin-1')

    return map(tensor, (x_train, y_train, x_valid, y_valid))