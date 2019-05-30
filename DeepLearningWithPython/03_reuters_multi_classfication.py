# @Time    : 2019/5/17 15:46
# @Author  : Leafage
# @File    : 03_reuters_multi_classfication.py
# @Software: PyCharm
# @Describe: retuers 数据集， 多分类任务
from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from keras import activations
import matplotlib.pyplot as plt


REUTERS_FILE_PATH = r'F:\GitRepository\MachineLearningNote\MachineLearningNote\DeepLearningWithPython\datasets\reuters.npz'
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(path=REUTERS_FILE_PATH, num_words=10000)

'''
# 把单词索引转换为单词文本
REUTERS_JSON = r'F:\GitRepository\MachineLearningNote\MachineLearningNote\DeepLearningWithPython\datasets\reuters_word_index.json'
word_index = reuters.get_word_index(path=REUTERS_JSON)
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_review)
'''


# 转变为one-hot 编码
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # 形状为len(sequences) * 10000 的零矩阵
    for i, sequence in enumerate(sequences):  # 返回list的索引位置和数据内容，这里的索引就是
        # i代表第i个评论，sequence对应的是评论中单词的索引， 将其设置为1，
        # 比如sequence=[1999,2222] ，则评论中包含的单词为对照表中的第1999和2222个两个单词
        # 所以将矩阵中的第i行中的第1999和2222列设置为1 ， 代表这两个单词出现了
        results[i, sequence] = 1.
    return results


# 数据one-hot向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# 把标签one-hot化
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


# 标签one-hot化
# one_hot_train_labels = to_one_hot(train_labels)
# one_hot_test_labels = to_one_hot(test_labels)


one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

print(train_labels[0])  # 3
print(one_hot_train_labels[0])  # [0,0,0,1,0,.....]


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation=activations.softmax))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 留出验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
# dict_keys(['val_loss', 'loss', 'val_acc', 'acc'])
print(history.history.keys())
# 绘画出训练损失和验证损失
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘画出训练精度和验证精度
plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 返回一个loss和一个metrics value
results = model.evaluate(x_test, one_hot_test_labels)

# predictions 为 预测值 ， 46维向量 ， 代表每个类别的概率
predictions = model.predict(x_test)
print(predictions[0])
print(np.sum(predictions[0]))
print(np.argmax(predictions[0]))

