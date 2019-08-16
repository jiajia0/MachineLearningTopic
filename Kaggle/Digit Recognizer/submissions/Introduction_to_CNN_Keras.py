# @Time    : 2019/8/16 12:10
# @Author  : Leafage
# @File    : Introduction_to_CNN_Keras.py
# @Software: PyCharm
# @Describe: 使用Keras CNN 进行识别。教程：https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white', context='notebook', palette='deep')

# 读取数据
train = pd.read_csv('../datasets/train.csv')
test = pd.read_csv('../datasets/test.csv')

# 得到训练数据
Y_train = train['label']
X_train = train.drop(labels=['label'], axis=1)

del train

# 观察结果分布
# g = sns.countplot(Y_train)
rs = Y_train.value_counts()
# plt.show()

# 查看空缺数据, isnull().any() 检查哪些列有空缺；describe()统计:count() 非空数值、unique：唯一数值、top：最高频、freq：最高频数
rs = X_train.isnull().any().describe()
rs = test.isnull().any().describe()

# 数据归一化
X_train = X_train / 255.0
test = test / 255.0

# 把数据reshape为：42000行，28*28*1
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

# 把label转变为one-hot编码
Y_train = to_categorical(Y_train, num_classes=10)

# 随机划分百分之十的数据作为验证集，剩下的作为训练集
random_seed = 2

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)

# 观察数据label
# plt.imshow(X_train[0][:,:,0])


# 构建模型
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))


