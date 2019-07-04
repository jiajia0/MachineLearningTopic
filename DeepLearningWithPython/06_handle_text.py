# @Time    : 2019/7/4 9:25
# @Author  : Leafage
# @File    : 06_handle_text.py
# @Software: PyCharm
# @Describe: 文本数据的预处理
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
tokenizer = Tokenizer(num_words=1000)  # 创建一个分词器，设置为只考虑前1000个最常见的单词
tokenizer.fit_on_texts(samples)  # 构建单词索引

# 这里得到的是单词对应的索引
sequences = tokenizer.texts_to_sequences(samples)  # 将字符串转换为整数索引组成的列表

# 这里已经把samples转换为one-hot编码了 , 若samples中单词出现在前1000个里面，就标记为1
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')  # 得到one-hot二进制表示

# 这里是一个dict ， k：word ， v：index
word_index = tokenizer.word_index  # 找回单词索引
