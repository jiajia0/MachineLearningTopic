# @Time    : 2018/1/11 12:58
# @Author  : Leafage
# @File    : OneR.py
# @Software: PyCharm
# @Describe: OneR算法实现，来自《Python数据挖掘入门与实践》

from collections import defaultdict
import operator
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np


def train_feature_value(dataSet, classify, feature_index, value):
    """
    根据传入的样本集和分类集，计算某个特征等于目标特征值属于哪一个分类
    :param dataSet: 数据集
    :param classify: 数据集对应的分类结果
    :param feature_index: 特征下标
    :param value: 特征值
    :return:
    """

    # 用来保存每个分类的个数
    class_counts = defaultdict(int)

    # 遍历所有的样本数据和对应的标签
    for sample, label in zip(dataSet, classify):
        # 如果当前样本的指定特征等于目标特征值的话
        if sample[feature_index] == value:
            # 对应的特征值个数加一
            class_counts[label] += 1

    # 根据分类的值排序，从大到小
    sorted_class_sounts = sorted(class_counts.items(), key=operator.itemgetter(1), reverse=True)

    # 找到对多的分类，也就是我们的目标分类
    most_class = sorted_class_sounts[0][0]

    # 在分类结果中找到分类不等于目标分类的
    incorrect_predictions = [class_count for class_value, class_count in class_counts.items()
                             if class_value != most_class]
    # 计算出错误个数
    error = sum(incorrect_predictions)

    return most_class, error


def train_on_feature(dataSet, classify, feature_index):
    """
    计算某一个特征的总错误率
    :param dataSet: 样本集
    :param classify: 样本集对应的分类结果
    :param feature_index: 特征下标
    :return:
    """

    # 得到该特征对应的所有可能性
    current_feature_full = set(dataSet[:, feature_index])

    # 用来存放当前特征的所有特征值的预测分类
    predictors = {}

    # 用来存放所有特征的错误个数
    errors = []

    # 遍历所有的特征可能性
    for current_feature in current_feature_full:
        # 计算当前特征值的预测分类和错误个数
        most_class, error = train_feature_value(dataSet, classify, feature_index, current_feature)
        predictors[current_feature] = most_class
        errors.append(error)

    # 计算出当前特征值分类的错误率
    error_chance = sum(errors)/float(len(dataSet))

    return predictors, error_chance


def classifyOneR(dataSet, classify):
    """
    实现OneR算法，找到特征错误率最低的特征作为划分结果
    :param dataSet: 样本集
    :param classify: 样本集对应的分类
    :return:
    """
    # 用来存放所有的预测特征
    all_predictors = {}

    # 用来存放所有预测特征的错误率
    errors = {}

    # 遍历所有的特征
    for feature_index in range(dataSet.shape[1]):
        # 使用当前特征进行预测
        predictors, total_error = train_on_feature(dataSet, classify, feature_index)

        # 存放当前的特征预测
        all_predictors[feature_index] = predictors

        # 存放当前的特征预测错误率
        errors[feature_index] = total_error

    # 找出错误率最低的一个
    print(errors)


def createDataSet():
    """
    创建测试的数据集
    :return:
    """
    # 加载数据集
    dataSet = load_iris()

    # 返回数据集的数据和分类结果
    return dataSet.data, dataSet.target


if __name__ == '__main__':
    dataSet, classify = createDataSet()
    attribute_means = dataSet.mean(axis=0)
    dataSet = np.array(dataSet >= attribute_means, dtype='int')
    trainData, testData, trainClassify, testClassify = train_test_split(dataSet, classify,
                                                                        test_size=0.25, random_state=14)

    classifyOneR(testData, testClassify)

