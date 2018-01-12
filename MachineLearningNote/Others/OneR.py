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
    根据传入的样本集和分类集，计算某个特征等于目标特征值时最可能属于哪一个分类，计算出错误个数
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
        # 将最可能划分的结果加入到预测结果中
        predictors[current_feature] = most_class
        errors.append(error)

    # 计算出当前特征值分类的错误率
    error_chance = sum(errors)/float(len(dataSet))

    return predictors, error_chance


def createModel(dataSet, classify):
    """
    实现OneR算法，找到特征错误率最低的特征作为划分结果，建立模型
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

    # 找出错误率最低的一个，用来划分特征
    best_feature, best_error = sorted(errors.items(), key=operator.itemgetter(1))[0]

    # 建立模型
    model = {'variable': best_feature, 'predictor': all_predictors[best_feature]}

    return model


def testOneR(testData, model, testClassify):
    """
    测试使用OneR实现的分类器
    :param testData: 测试数据集
    :param model: 建立的模型
    :param testClassify: 测试数据集对应的分类
    :return:
    """
    # 拿到模型中用来分类的特征下标
    variable = model['variable']

    # 拿到特征对应的划分集
    predictor = model['predictor']

    # 对测试集进行分类
    predicted = np.array([predictor[int(sample[variable])] for sample in testData])

    # 计算错误率
    accuracy = np.mean(predicted == testClassify) * 100

    print(accuracy)

    return accuracy


def classifyOneR(inX, model):
    """
    使用OneR进行预测分类
    :param inX: 要进行预测的数据
    :param model: 建立的模型
    :return:
    """
    # 拿到模型中用来分类的特征下标
    variable = model['variable']

    # 拿到特征对应的划分集
    predictor = model['predictor']

    # 进行分类
    classify = predictor[inX[variable]]

    return classify


def createDataSet():
    """
    创建测试的数据集
    :return:
    """
    # 加载数据集
    dataSet = load_iris()

    # 返回数据集的数据和分类结果
    return dataSet.data, dataSet.target


def createDataSetXiGua():
    """
    创建西瓜书中的数据集
    :return:
    """
    dataSet = [
        # 1
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 2
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        # 3
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 4
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        # 5
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 6
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        # 7
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        # 8
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],

        # ----------------------------------------------------
        # 9
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
        # 10
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
        # 11
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
        # 12
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
        # 13
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
        # 14
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        # 15
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        # 16
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        # 17
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
    ]

    # 特征值列表
    labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感']

    dataSet = np.array(dataSet)

    # 得到分类结果
    classify = dataSet[:, -1]

    # 去除分类之后的数据集
    dataSet = dataSet[:, :-1]

    return dataSet, classify, labels


def createDataSetForPerson():
    """
    使用身高、眼睛大小、肤色作为数据集
    :return:
    """
    dataSet = [
        # 1
        [1, '正常', '偏白', '男'],
        # 2
        [1, '较大', '偏黑', '男'],
        # 3
        [1, '正常', '偏黑', '男'],
        # 4
        [1, '较大', '偏黑', '女'],
        # 5
        [0, '正常', '偏白', '女'],
        # 6
        [0, '较大', '偏白', '女'],
    ]

    # 特征值列表
    labels = ['身高', '眼睛大小', '肤色']

    dataSet = np.array(dataSet)

    # 得到分类结果
    classify = dataSet[:, -1]

    # 去除分类之后的数据集
    dataSet = dataSet[:, :-1]

    return dataSet, classify, labels


if __name__ == '__main__':
    dataSet, classify, labels = createDataSetForPerson()
    model = createModel(dataSet, classify)
    newData = np.array(['1', '正常', '偏黑'])
    result = classifyOneR(newData, model)
    print(result)

    # # ----------使用西瓜书中的数据集----------
    # dataSet, classify, labels = createDataSetXiGua()
    #
    # model = createModel(dataSet, classify)
    #
    # result = classifyOneR(dataSet[0], model)
    # print(dataSet[0])
    # print(result)

    # ----------使用sklear中的数据集----------
    # dataSet, classify = createDataSet()
    # attribute_means = dataSet.mean(axis=0)
    # dataSet = np.array(dataSet >= attribute_means, dtype='int')
    # trainData, testData, trainClassify, testClassify = train_test_split(dataSet, classify,
    #                                                                     test_size=0.25, random_state=14)
    # model = createModel(trainData, trainClassify)
    #
    # testOneR(testData, model, testClassify)
    #
    # result = classifyOneR(testData[0], model)
    # print(result)
    # print(testClassify[0])
