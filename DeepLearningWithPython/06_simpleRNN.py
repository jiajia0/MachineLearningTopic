# @Time    : 2019/7/4 11:48
# @Author  : Leafage
# @File    : 06_simpleRNN.py
# @Software: PyCharm
# @Describe: 简单的RNN的Numpy实现
import numpy as np

timesteps = 100  # 输入序列的时间步数
input_features = 32  # 输入特征空间的维度
output_features = 64  # 输出特征空间的维度

# 输入数据， 随机生成， 仅作示例
inputs = np.random.random((timesteps, input_features))

# 初始状态：全零向量
state_t = np.zeros((output_features))

# 创建随机的权重
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features, ))

print(inputs.shape)
print(W.shape)

successive_outputs = []
for input_t in inputs:
    # W: 64 * 32, input_t: 32, -> output_t: 64,
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)  # 由输入和当前状态（前一个输出）计算得到当前输出
    successive_outputs.append(output_t)
    state_t = output_t
# 一共有timesteps个output_features
final_output_sequence = np.stack(successive_outputs, axis=0)
