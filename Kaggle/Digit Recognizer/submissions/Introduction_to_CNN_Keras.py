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
from Utils.plot_confusion_matrix import plot_confusion_matrix

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

# 把数据reshape为：42000行，28*28*1 , 第一个参数-1为：自动计算出有多少个28*28*1
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

# 把label转变为one-hot编码
Y_train = to_categorical(Y_train, num_classes=10)

# 随机划分百分之十的数据作为验证集，剩下的作为训练集
random_seed = 2

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)

# 观察数据label , 28*28 ,二维图
# plt.imshow(X_train[0][:,:,0])
# print(X_train[0].shape)  # 28*28*1
# plt.show()

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

# 设置优化器
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 学习速率调整
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
epochs = 30
batch_size = 86

# 数据增强
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                              epochs=epochs, validation_data=(X_val, Y_val),
                              verbose=2, steps_per_epoch=X_train.shape[0] // batch_size,
                              callbacks=[learning_rate_reduction])

# 画出训练集和测试集的loss和accuracy
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label='Training loss')
ax[0].plot(history.history['val_loss'], color='r', label='validation loss', axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label='Training accuracy')
ax[1].plot(history.history['val_acc'], color='r', label='Validation accuracy')
legend = ax[1].legend(loc='best', shadow=True)

plt.show()

Y_pred = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_val, axis=1)
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
plot_confusion_matrix(confusion_mtx, classes=range(10))

# 绘制出错误的一些分类观察
errors = (Y_pred_classes - Y_true != 0)  # 找到错误的分类
Y_pred_classes_errors = Y_pred_classes[errors]  # 选择出错误的分类标签
Y_pred_errors = Y_pred[errors]  # 预测标签：one-hot类型
Y_true_errors = Y_true[errors]  # 错误标签对应的正确标签
X_val_errors = X_val[errors]  # 错误标签原来数组


def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow((img_errors[error]).reshape((28, 28)))
            ax[row, col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error], obs_errors[error]))
            n += 1
    plt.show()


# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors, axis=1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
