# @Time    : 2019/6/9 10:16
# @Author  : Leafage
# @File    : 05_convnet_mnist.py
# @Software: PyCharm
# @Describe: 利用卷积神经网络识别mnist数据集，准确率为98%以上
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

MNIST_FILE_PATH = r'F:\GitRepository\MachineLearningNote\DeepLearningWithPython\datasets\mnist.npz'
(train_images, train_labels), (test_images, test_labels) = mnist.load_data(MNIST_FILE_PATH)

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 搭建神经网络
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 训练model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# 在模型上进行评估
test_loss, test_acc = model.evaluate(test_images, test_labels)