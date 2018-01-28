# @Time    : 2018/1/28 13:34
# @Author  : Leafage
# @File    : sklearn_knn_ionosphere.py
# @Software: PyCharm
# @Describe: 使用电离层的数据，利用sklearn中的KNN分类器进行分类，该内容来自《Python数据挖掘入门与实践》


import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


def loadDataSet():
    """
    加载数据集
    :return:
    """
    # 创建一个与数据集相等的矩阵，不包含标签
    x = np.zeros((351, 34), dtype=float)
    # 用来保存标签
    y = np.zeros((351, ), dtype=bool)
    # 加载数据集
    reader = csv.reader(open('ionosphere.data'))

    # 同时获取索引和值
    for i, row in enumerate(reader):
        # 将数据添加到x中
        data = [float(datum) for datum in row[:-1]]
        x[i] = data
        # 等于g设置为1，否则设置为0
        y[i] = row[-1] == 'g'

    return x, y


def split_dataset_test(dataset, labels):
    """
    手动对整个数据集进行划分测试
    :param dataset 整个数据集
    :param labels 数据集对应的标签
    :return:
    """
    # 划分训练集和测试数据集，不设置test_size的话默认的为0.25
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels, random_state=14)
    # 创建一个kNN分类器
    estimator = KNeighborsClassifier()
    # 使用训练集进行训练
    estimator.fit(x_train, y_train)
    # 使用测试集进行测试算法
    y_predicted = estimator.predict(x_test)
    # 计算正确率
    accuracy = np.mean(y_test == y_predicted) * 100
    print('使用划分数据集进行训练的正确率为：{0: .1f}%'.format(accuracy))


def val_score(dataSet, labels):
    """
    使用sklearn中的交叉验证计算正确率
    :param dataSet:
    :param labels:
    :return:
    """
    estimator = KNeighborsClassifier()
    scores = cross_val_score(estimator, dataSet, labels, scoring='accuracy')
    average_accuracy = np.mean(scores) * 100
    print('使用交叉验证得到的准确率为：{0: .1f}%'.format(average_accuracy))


def test_n_neighbors(dataSet, labels):
    """
    使用不同的k值进行计算准确率
    :param dataSet 数据集
    :param labels 对应的标签
    :return:
    """
    # 用来保存平均准确率
    avg_scores = []
    # 全部的准确率
    all_scores = []
    # 设定k的值从1到20
    parameter_values = list(range(1, 21))
    # 对每个k值的准确率进行计算
    for n_neighbors in parameter_values:
        # 创建KNN分类器
        estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_val_score(estimator, dataSet, labels, scoring='accuracy')
        avg_scores.append(np.mean(scores))
        all_scores.append(scores)

    plt.plot(parameter_values, avg_scores, '-o')
    plt.show()


def test_autoNorm(dataSet, labels):
    """
    将数据进行归一化
    :param dataSet: 原始数据
    :param labels: 对应的标签
    :return:
    """
    # 创建一个副本
    x_broken = np.array(dataSet)
    # 每隔一列将特征除以10
    x_broken[:, ::2] /= 10
    estimator = KNeighborsClassifier()
    # 计算原始数据的准确率
    original_scores = cross_val_score(estimator, dataSet, labels, scoring='accuracy')
    print('原始数据的准确率为：{0: .1f}%'.format(np.mean(original_scores) * 100))
    # 计算破坏数据及之后的准确率
    broken_scores = cross_val_score(estimator, x_broken, labels, scoring='accuracy')
    print('破坏数据集之后的准确率为：{0: .1f}%'.format(np.mean(broken_scores) * 100))

    # 然后将其归一化
    x_transformed = MinMaxScaler().fit_transform(x_broken)
    # 再次计算准确率
    transformed_scores = cross_val_score(estimator, x_transformed, labels, scoring='accuracy')
    print('数据归一化后的准确率为：{0: .1f}%'.format(np.mean(transformed_scores) * 100))


def test_pipeline(dataSet, labels):
    """
    使用sklearn中的流水线步骤进行计算
    :param dataSet:
    :param labels:
    :return:
    """
    # 创建一个副本
    x_broken = np.array(dataSet)
    # 每隔一列将特征除以10
    x_broken[:, ::2] /= 10
    # 创建流水线
    scaling_pipeline = Pipeline([('scale', MinMaxScaler()), ('predict', KNeighborsClassifier())])
    scores = cross_val_score(scaling_pipeline, x_broken, labels, scoring='accuracy')
    print('使用流水线结构的准确率为：{0: .1f}%'.format(np.mean(scores) * 100))


if __name__ == '__main__':
    dataSet, labels = loadDataSet()
    # 手动划分训练集与测试集进行预测
    # split_dataset_test(dataSet, labels)
    # 使用sklearn中的交叉验证测试正确率
    # val_score(dataSet, labels)
    # 使用不同的k值计算准确率
    # test_n_neighbors(dataSet, labels)
    # 预处理数据，将其归一化并计算准确率
    # test_autoNorm(dataSet, labels)
    # 使用流水线的工作结构
    test_pipeline(dataSet, labels)
