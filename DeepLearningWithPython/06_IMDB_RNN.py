# @Time    : 2019/7/5 9:15
# @Author  : Leafage
# @File    : 06_IMDB_RNN.py
# @Software: PyCharm
# @Describe: 使用RMM进行IMDB评论分类
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, SimpleRNN
from keras.models import Sequential
from keras import optimizers, losses
import matplotlib.pyplot as plt


IMDB_FILE_PATH = r'F:\GitRepository\MachineLearningNote\DeepLearningWithPython\datasets\imdb.npz'
max_features = 10000  # 只使用前10000个最常见的单词
maxlen = 500  # 在这么多单词之后截断文本（这些单词都属于前max_features个最常见的单词）
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(path=IMDB_FILE_PATH, num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples * time)')
# 把数据转换为(samples x maxlen)的二维向量 , 长度不够500的序列，前面使用0进行填充
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)


# 构建模型
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss=losses.binary_crossentropy, metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)  # 指定20%的数据作为验证集，验证数据不参与训练

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
