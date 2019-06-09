# @Time    : 2019/4/26 20:55
# @Author  : Leafage
# @File    : 02_example.py
# @Software: PyCharm
# @Describe: 第二章中的例子
from keras.datasets import mnist
import matplotlib.pylab as plt
import numpy as np
from keras import models
from keras import layers
from keras.utils import to_categorical


MNIST_FILE_PATH = r'F:\GitRepository\MachineLearningNote\DeepLearningWithPython\datasets\mnist.npz'
(train_images, train_labels), (test_images, test_labels) = mnist.load_data(MNIST_FILE_PATH)
# digit = train_images[4]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

# 定义网络
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# 编译网络
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 拟合模型
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 计算精度
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc', test_acc)