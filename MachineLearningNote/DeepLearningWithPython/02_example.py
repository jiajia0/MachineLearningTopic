# @Time    : 2019/4/26 20:55
# @Author  : Leafage
# @File    : 02_example.py
# @Software: PyCharm
# @Describe: 第二章中的例子
from keras.datasets import mnist
import matplotlib.pylab as plt
import numpy as np


MNIST_FILE_PATH = r'F:\GitRepository\MachineLearningNote\MachineLearningNote\DeepLearningWithPython\datasets\mnist.npz'
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data(MNIST_FILE_PATH)
# digit = train_images[4]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

data = np.load(MNIST_FILE_PATH)
print(data['x_test'].shape)
print(data['x_train'].shape)
print(data['y_test'].shape)
print(data['y_train'].shape)