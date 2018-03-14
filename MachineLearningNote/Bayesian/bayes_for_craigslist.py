# @Time    : 2018/3/14 17:05
# @Author  : Leafage
# @File    : bayes_for_craigslist.py
# @Software: PyCharm
# @Describe: 《机器学习实战》贝叶斯部分关于Craigslist言语分析
import feedparser
import operator
from bayes import *
import re
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


def calcostFreq(vocabList, fullText):
    """
    统计文本中出现频率最高的30个单词
    :param vocabList: 单词集合
    :param fullText: 文本
    :return:
    """
    # 用来保存每个单词出现的次数
    freqDict = {}
    # 循环遍历所有的单词
    for token in vocabList:
        # 计算该单词在文本中出现的次数
        freqDict[token] = fullText.count(token)
    # 按照value进行排序
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    # 返回频率最高的前30个单词
    return sortedFreq[: 30]


def localWords(feed1, feed0):
    """
    使用贝叶斯来预测出两个RSS源中的摘要部分单词出现的频率
    :param feed1: 第一个RSS源，分类时使用1来表示
    :param feed0:第二个RSS源，分类时使用0来表示
    :return:
    """
    # 用来保存RSS源中摘要部分的单词列表，每个摘要对应一个列表
    docList = []
    # 用来保存RSS源中单词列表是属于哪一个分类的，feed1用1来表示，feed0用0表示
    classList = []
    # 保存两个RSS源中的单词
    fullText = []
    # 计算出两个文档中哪一个文档的数目少，防止索引越界
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    # 在合适范围内遍历RSS信息
    for i in range(minLen):
        # 将feed1对应的RSS源中的摘要部分拆分为单词，并且保存
        wordList = textParse(feed1['entries'][i]['summary'])
        # 将该单词列表添加到列表中
        docList.append(wordList)
        # 添加到总的列表中，使用扩展的方式
        fullText.extend(wordList)
        # fee1用1来表示
        classList.append(1)
        # 将feed0对应的RSS源中的摘要部分拆分为单词，并且保存
        wordList = textParse(feed0['entries'][i]['summary'])
        # 将该单词列表添加到列表中
        docList.append(wordList)
        # 添加到总的列表中，使用扩展的方式
        fullText.extend(wordList)
        # fee0用0来表示
        classList.append(0)
    # 将所有的单词去重，用一个列表示去重后的单词
    vocabList = createVocabList(docList)
    # 计算出频率最多的30个单词
    top30Words = calcostFreq(vocabList, fullText)

    # 移除这30个频率最高的单词，因为这些单词属于辅助词
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    # 用来随机选取索引
    trainingSet = list(range(2 * minLen))
    # 用来保存随机选取的测试数据的索引
    testSet = []
    # 随机选取20个数据
    for i in range(20):
        # 随机生成一个索引值
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 将其添加到测试数据索引中
        testSet.append(trainingSet[randIndex])
        # 删除该索引，防止再次出现
        del(trainingSet[randIndex])

    # 用来保存训练数据，也就是总的数目减去刚才选取的20份所剩下的
    trainMat = []
    # 训练数据对应的分类
    trainClasses = []
    # 遍历所有的随机索引
    for docIndex in trainingSet:
        # 统计来列表中各个单词出现的次数并添加到列表中
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        # 对应的分类
        trainClasses.append(classList[docIndex])

    # 计算出各个单词出现的概率，p0V就是feed0中各个单词在feed0分类下出现的概率，pSpam就是feed1的先验概率
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    # 用来统计分类错误的个数
    errorCount = 0
    # 将刚才选取的20个数据进行预测
    for docIndex in testSet:
        # 统计所要分类的文档中的各个单词出现的次数
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        # 使用贝叶斯分类器进行分类，并统计下不正确的个数
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1

    print('错误率为：%f' % float(errorCount/len(testSet)))
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    """
    得到最具代表性的词汇
    :param ny:
    :param sf:
    :return:
    """
    # 得到单词表以及ny中的单词在ny中出现的概率p1V，sf出现的概率P0V
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []

    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=operator.itemgetter(1), reverse=True)
    sortedNY = sorted(topNY, key=operator.itemgetter(1), reverse=True)
    print('-------------------SF--------------------------')
    for item in sortedSF:
        print(item)
    print('-------------------NY--------------------------')
    for item in sortedNY:
        print(item)


if __name__ == '__main__':
    feed1 = feedparser.parse('https://newyork.craigslist.org/stp/index.rss')
    feed0 = feedparser.parse('https://sfbay.craigslist.org/stp/index.rss')
    getTopWords(feed1, feed0)

