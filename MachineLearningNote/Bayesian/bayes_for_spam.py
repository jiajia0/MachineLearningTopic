# @Time    : 2018/3/9 19:57
# @Author  : Leafage
# @File    : bayes_for_spam.py
# @Software: PyCharm
# @Describe:  《机器学习实战》贝叶斯部分关于垃圾邮件过滤实现
import re
from bayes import *
import random


def textParse(bigString):
    """
    将一个字符串转换为字符串列表，去掉小于两个字母的单词，并且转换为小写
    :param bigString:
    :return:
    """
    # \W 使用非字母作为分隔符
    listOftokens = re.split(r'\W*', bigString)
    # 排除长度小于2的单词，并且转换为小写
    return [tok.lower() for tok in listOftokens if len(tok) > 2]


def spamTest():
    """
    使用贝叶斯分类器完成垃圾邮件分类测试，使用交叉验证方法
    :return:
    """
    # 用来保存电子邮件的单词列表，每个电子邮件对应一个单词列表
    docList = []
    # 用来保存对应的电子邮件的分类
    classList = []
    # 用来保存所有的单词
    fullText = []

    # 测试数据只有25个
    for i in range(1, 26):
        # 读取对应的电子邮件，并且转换为单词列表、垃圾电子邮件
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        # 1代表是垃圾邮件
        classList.append(1)

        # 读取对应的电子邮件，并且转换为单词列表、正常电子邮件
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        # 0代表是正常邮件（非垃圾邮件）
        classList.append(0)

    # 将所有的单词去重，生成一个总的词汇表
    vocabList = createVocabList(docList)
    # 代表着50份邮件的下标
    trainingSet = list(range(50))
    # 用来存放随机选取的邮件下标
    testSet = []

    # 随机选取十份电子邮件进行测试
    for i in range(10):
        # 随机生成索引
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 将对应的索引添加到列表中
        testSet.append(trainingSet[randIndex])
        # 删除该索引防止再次出现
        del trainingSet[randIndex]

    # 用来存放测试数据矩阵
    trainMat= []
    # 用来存放测试数据的分类
    trainClasses = []
    # 将刚才随机生成的十份电子邮件分别处理
    for docIndex in trainingSet:
        # 处理对应索引的电子邮件，将其转换为一个列表，列表长度为词汇表的长度，0代表该单词在当前邮件中没有出现，1代表出现了
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        # 当前电子邮件的对应分类
        trainClasses.append(classList[docIndex])

    # 计算出单词出现的频率，p0V是正常电子邮件，p1V是垃圾邮件,pSpam是垃圾邮件的先验概率（也就是垃圾邮件的数目除以总的邮件数目）
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))

    # 用来标记使用贝叶斯分类错误的个数
    errorCount = 0
    # 分类对刚才随机抽取的十份电子邮件进行分类
    for docIndex in testSet:
        # 查看总词汇中哪个单词在该邮件中出现了，生成一个列表，0没有出现，1出现
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # 如果分类错误，则错误个数加一
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1

    print('使用贝叶斯分类器，错误的概率为：' + str(float(errorCount/len(testSet))))


if __name__ == '__main__':
    spamTest()

