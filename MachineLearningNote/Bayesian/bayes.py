# @Time    : 2018/1/6 18:37
# @Author  : Leafage
# @File    : bayes.py
# @Software: PyCharm
# 基于《机器学习实战》的贝叶斯部分代码
from numpy import *


def loadDataSet():
    """
    加载数据集
    :return:
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'mh', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    # 1代表侮辱性文字，0代表正常言论
    classVec = [0, 1, 0, 1, 0, 1]

    return postingList, classVec


def createVocabList(dataSet):
    """
    创建一个包含在所有文档中出现的不重复词的列表
    :param dataSet: 文档
    :return: 去重之后的词列表
    """
    # 用来存放不重复词的集合
    vocabSet = set([])

    # 遍历整个数据集文档
    for document in dataSet:
        # 如果当前的词不存在集合中则将其添加进去
        vocabSet = vocabSet | set(document)

    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    用来检测文档中是否出现词汇表中的数据，0代表没有出现，1代表出现
    :param vocabList: 词汇表
    :param inputSet:  文档
    :return:
    """
    # 首先创建一个全是0的列表，列表长度就是词汇表的长度
    returnVec = [0] * len(vocabList)

    # 遍历一遍文档
    for word in inputSet:
        # 如果文档中的某个单词在词汇表中，则将其设置为1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1

    return returnVec


def bagOfWords2Vec(vocabList, inputSet):
    """
    用来检测文档中是否出现词汇表中的数据，0代表没有出现，没出现一次就加一
    :param vocabList: 词汇表
    :param inputSet:  文档
    :return:
    """
    # 首先创建一个全是0的列表，列表长度就是词汇表的长度
    returnVec = [0] * len(vocabList)

    # 遍历一遍文档
    for word in inputSet:
        # 如果文档中的某个单词在词汇表中，则将加1
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1

    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    计算出每个单词在各类文档中出现的概率
    :param trainMatrix: 文档矩阵
    :param trainCategory: 每个文档对应的类别，1是侮辱性的，0是非侮辱性的
    :return:
    """
    # 计算出文档的总数
    numTrainDocs = len(trainMatrix)

    # 计算出每个文档有多少词汇
    numWords = len(trainMatrix[0])

    # 计算出属于侮辱类的概率，也就是先验概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)

    # 初始化为0，用来标记每个单词出现的次数
    p0Num = ones(numWords)
    p1Num = ones(numWords)

    # 用来记录分母
    p0Denom = 2.0
    p1Denom = 2.0

    # 遍历所有文档
    for i in range(numTrainDocs):
        # 如果当前的文档时具有侮辱性
        if trainCategory[i] == 1:
            # 计算出每个单词出现的次数
            p1Num += trainMatrix[i]
            # 计算出侮辱性类别的总的单词数
            p1Denom += sum(trainMatrix[i])
        # 如果不具有侮辱性
        else:
            # 计算出每个单词的出现次数
            p0Num += trainMatrix[i]
            # 计算出非侮辱性类别的总的单词书
            p0Denom += sum(trainMatrix[i])

    # 计算侮辱性的每个单词出现的概率，取对数防止溢出
    p1Vect = log(p1Num/p1Denom)
    # 计算非侮辱性的每隔单词出现的概率
    p0Vect = log(p0Num/p0Denom)

    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    朴素贝叶斯分类函数
    :param vec2Classify: 需要分类的数据
    :param p0Vec: 非侮辱类中的各个单词的条件概率
    :param p1Vec: 侮辱类中的各个单词的条件概率
    :param pClass1:侮辱类的先验概率
    :return: 返回分类结果
    """

    # 计算出测试数据中出现的单词在样本中的贝叶斯概率，logab = loga + logb
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)

    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0Vect, p1Vect, pAbusive = trainNB0(trainMat, listClasses)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    classify = classifyNB(thisDoc, p0Vect, p1Vect, pAbusive)

    if classify:
        print('侮辱类')
    else:
        print('非侮辱类')

