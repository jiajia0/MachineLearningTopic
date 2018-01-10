# @Time    : 2018/1/9 15:16
# @Author  : Leafage
# @File    : xigua_bayes.py
# @Software: PyCharm
# 西瓜书中的贝叶斯实现
from math import *
import numpy as np
import operator


def calcDisperseChance(dataSet, axis, value, classify):
    """
    计算离散属性的特征值概率，在分类等于classify的情况下，样本数据集dataSet中axis下标对应的特征值等于value的概率
    :param dataSet: 样本数据集
    :param axis: 特征值的下标
    :param value: 对应的特征值的值
    :param classify: 对应的分类
    :return: 返回概率
    """
    # 用来存放数据集中分类为classify的内容
    classifySet = []

    # 循环遍历所有的数据集，找到分类为classify的内容
    for i in range(len(dataSet)):
        if dataSet[i][-1] == classify:
            classifySet.append(dataSet[i])

    # 用来存放已知类别中特征值等于value的内容
    labelClassifySet = []

    # 循环所有的分类数据集，找到特征值等于value的内容
    for i in range(len(classifySet)):
        if classifySet[i][axis] == value:
            labelClassifySet.append(classifySet[i])

    return len(labelClassifySet)/float(len(classifySet))


def calcSeriesChance(dataSet, axis, value, classify):
    """
    计算样本集中的连续值概率，利用高斯分布进行计算
    :param dataSet: 样本集
    :param axis: 特征值对应的下标
    :param value: 测试样本的对应特征的值
    :param classify: 分类
    :return: 概率
    """
    # 用来存放数据集中分类为classify的内容
    classifySet = []

    # 循环遍历所有的数据集，找到分类为classify的内容
    for i in range(len(dataSet)):
        if dataSet[i][-1] == classify:
            classifySet.append(dataSet[i])

    # 使用numpy存储特征值数据，方便计算
    n_classifySet = np.array([example[axis] for example in classifySet])

    # 计算对应特征值的均值
    mean = n_classifySet.mean()

    # 计算特征值的标准差
    std = np.std(n_classifySet)

    # 计算出概率，根据概率密度函数公式进行计算
    chance = (1/(sqrt(2*pi)*std))*(exp(-(pow((value-mean), 2)/(2*pow(std, 2)))))

    return chance


def bayesClassify(dataSet, testSet):
    """
    使用贝叶斯公式进行计算分类
    :param dataSet: 样本数据集
    :param testSet: 测试数据集
    :return: 返回分类结果
    """
    # 找到样本集中的所有分类
    classifys = set([example[-1] for example in dataSet])

    # 测试数据及属于某个分类的概率
    classifyChance = {}

    # 遍历所有的分类
    for classify in classifys:

        # 概率初始化为当前分类所占的比例，也就是先验概率
        chance = len([example[-1] for example in dataSet if example[-1] == classify])/float(len(dataSet))

        # 遍历所有的测试数据集，计算对应分类的概率
        for i in range(len(testSet)):
            if isinstance(testSet[i], str):
                # 计算当前分类下的特征的概率
                chance *= calcDisperseChance(dataSet, i, testSet[i], classify)
            if isinstance(testSet[i], float):
                chance *= calcSeriesChance(dataSet, i, testSet[i], classify)
        classifyChance[classify] = chance

    # 根据value进行排序
    sortedChance = sorted(classifyChance.items(), key=operator.itemgetter(1), reverse=True)
    return sortedChance


def createDataSet():
    """
    创建测试的数据集，里面的数值中具有连续值
    :return:
    """
    dataSet = [
        # 1
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
        # 2
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
        # 3
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
        # 4
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
        # 5
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
        # 6
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
        # 7
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
        # 8
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],

        # ----------------------------------------------------
        # 9
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
        # 10
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
        # 11
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
        # 12
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
        # 13
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
        # 14
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
        # 15
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
        # 16
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
        # 17
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']
    ]
    # 特征值列表
    labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']

    # 特征对应的所有可能的情况
    labels_full = {}

    for i in range(len(labels)):
        labelList = [example[i] for example in dataSet]
        uniqueLabel = set(labelList)
        labels_full[labels[i]] = uniqueLabel

    return dataSet, labels, labels_full


if __name__ == '__main__':
    dataSet, labels, labels_full = createDataSet()
    # 测试数据，需要进行预测
    testSet = ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.243, 0.105]
    classify = bayesClassify(dataSet, testSet)
    print(classify)
