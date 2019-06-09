# @Time    : 2019/5/19 14:57
# @Author  : Leafage
# @File    : 03_boston_regression.py
# @Software: PyCharm
# @Describe: 波士顿房价预测回归
from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

BOSTON_FILE_PATH = r'F:\GitRepository\MachineLearningNote\DeepLearningWithPython\datasets\boston_housing.npz'
(train_data, train_targets), (test_data, testa_targets) = boston_housing.load_data(path=BOSTON_FILE_PATH)


# 对数据标准化，每个特征值减去平均特征值，再除以标准差，这样的特征值平均值为0，标准差为1
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

# 都是使用的训练集的mean和std
test_data -= mean
test_data /= std


# 构建网络
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    # mse 均方误差 : mean absolute error , mase 平均绝对误差 ：差的绝对值
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# k 折验证
k = 4
# 划分为k个区域，每个区域的数量
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
# 进行k次
for i in range(k):
    print('processing fold # ', i)
    # 准备验证数据，第k个分区的数据
    val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]
    # 准备训练数据
    partial_train_data = np.concatenate([train_data[: i * num_val_samples],
                                         train_data[(i+1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate([train_targets[: i * num_val_samples],
                                            train_targets[(i+1) * num_val_samples:]],
                                           axis=0)

    model = build_model()  # 构建网络，已经编译过的
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))

num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    # 验证数据
    val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    # 其他的数据作为训练数据
    partial_train_data = np.concatenate([train_data[: i * num_val_samples],
                                         train_data[(i+1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate([train_targets[: i * num_val_samples],
                                            train_targets[(i+1) * num_val_samples:]],
                                           axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

# 计算所有伦次中的K折验证分数平均值
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# 删除前10个数据点，并且每个数据点替换为前面数据点的指数移动平均值，得到光滑的曲线
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


smooth_mase_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mase_history) + 1), smooth_mase_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()