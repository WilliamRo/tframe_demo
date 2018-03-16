import os

from tframe import TFData
from tframe import pedia
from tframe.utils.tfdata import load_cifar10
from tframe.utils.tfdata import load_cifar10tfd


DATA_DIR = r'../data/CIFAR-10/'
# DATA_PATH = r'../data/CIFAR-10/cifar-10-test.tfd'


CIFAR10 = load_cifar10(DATA_DIR, one_hot=True, validation_size=0)
CIFAR10[pedia.training].save(os.path.join(DATA_DIR, 'cifar-10-train.tfd'))
CIFAR10[pedia.test].save(os.path.join(DATA_DIR, 'cifar-10-test.tfd'))
# data = TFData.load(DATA_PATH)
# data = load_cifar10tfd(DATA_DIR, validation_size=10000)

# path = __file__
# print(path)
# path = os.path.dirname(path)
# print(path)
# path = os.path.dirname(path)
# print(path)

