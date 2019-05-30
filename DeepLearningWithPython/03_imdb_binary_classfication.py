# @Time    : 2019/5/16 14:51
# @Author  : Leafage
# @File    : 03_imdb_binary_classfication.py
# @Software: PyCharm
# @Describe: imdb 电影评论二分类
from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt


# 使用单词频率最大的前10000个数据
IMDB_FILE_PATH = r'F:\GitRepository\MachineLearningNote\MachineLearningNote\DeepLearningWithPython\datasets\imdb.npz'
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(path=IMDB_FILE_PATH, num_words=10000)

# print(max([max(sequence) for sequence in train_data])) # 每个train_data[i] 代表一个评论，其中存放的是单词的索引，最大的索引不超过10000
# 对应的train_labels 是这条评论是正面的还是负面的， 0 负面 1 正面

# 利用索引与单词的对照表，可以将评论解析为原来的句子
# WORD_INDEX_PATH = r'F:\GitRepository\MachineLearningNote\MachineLearningNote\DeepLearningWithPython\datasets\imdb_word_index.json'
# word_index = imdb.get_word_index(path=WORD_INDEX_PATH)  # 单词与索引的对照表
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# 转变为one-hot 编码
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # 形状为len(sequences) * 10000 的零矩阵
    for i, sequence in enumerate(sequences):  # 返回list的索引位置和数据内容，这里的索引就是
        # i代表第i个评论，sequence对应的是评论中单词的索引， 将其设置为1，
        # 比如sequence=[1999,2222] ，则评论中包含的单词为对照表中的第1999和2222个两个单词
        # 所以将矩阵中的第i行中的第1999和2222列设置为1 ， 代表这两个单词出现了
        results[i, sequence] = 1.
    return results


# 训练数据 one-hot 化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

# 留出验证集
x_val = x_train[: 10000]
partial_x_train = x_train[10000:]
y_val = y_train[: 10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
history_dict = history.history
# print(history_dict.keys())  # ['val_loss', 'val_binary_accuracy', 'loss', 'binary_accuracy']

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)  # 训练的次数

# loss 可视化
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# acc 可视化
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()