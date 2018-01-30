# @Time    : 2017/12/17 17:25
# @Author  : Leafage
# @File    : trees.py
# @Software: PyCharm

from math import log
import collections
import operator
from xigua import treePlotter


def calcShannonEnt(dataSet):
    """
    计算给定数据集的信息熵(香农熵)
    :param dataSet:
    :return:
    """
    # 计算出数据集的总数
    numEntries = len(dataSet)

    # 用来统计标签
    labelCounts = collections.defaultdict(int)

    # 循环整个数据集，得到数据的分类标签
    for featVec in dataSet:
        # 得到当前的标签
        currentLabel = featVec[-1]

        # # 如果当前的标签不再标签集中，就添加进去（书中的写法）
        # if currentLabel not in labelCounts.keys():
        #     labelCounts[currentLabel] = 0
        #
        # # 标签集中的对应标签数目加一
        # labelCounts[currentLabel] += 1

        # 也可以写成如下
        labelCounts[currentLabel] += 1

    # 默认的信息熵
    shannonEnt = 0.0

    for key in labelCounts:
        # 计算出当前分类标签占总标签的比例数
        prob = float(labelCounts[key]) / numEntries

        # 以2为底求对数
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    按照给定的特征值，将数据集划分
    :param dataSet: 数据集
    :param axis: 给定特征值的坐标
    :param value: 给定特征值满足的条件，只有给定特征值等于这个value的时候才会返回
    :return:
    """
    # 创建一个新的列表，防止对原来的列表进行修改
    retDataSet = []

    # 遍历整个数据集
    for featVec in dataSet:
        # 如果给定特征值等于想要的特征值
        if featVec[axis] == value:
            # 将该特征值前面的内容保存起来
            reducedFeatVec = featVec[:axis]
            # 将该特征值后面的内容保存起来，所以将给定特征值给去掉了
            reducedFeatVec.extend(featVec[axis + 1:])
            # 添加到返回列表中
            retDataSet.append(reducedFeatVec)

    return retDataSet


def chooseBestFeatureToSplit(dataSet, labels):
    """
    选择最好的数据集划分特征，根据信息增益值来计算
    :param dataSet:
    :return:
    """
    # 得到数据的特征值总数
    numFeatures = len(dataSet[0]) - 1

    # 计算出基础信息熵
    baseEntropy = calcShannonEnt(dataSet)

    # 基础信息增益为0.0
    bestInfoGain = 0.0

    # 最好的特征值
    bestFeature = -1

    # 对每个特征值进行求信息熵
    for i in range(numFeatures):
        # 得到数据集中所有的当前特征值列表
        featList = [example[i] for example in dataSet]

        # 将当前特征唯一化，也就是说当前特征值中共有多少种
        uniqueVals = set(featList)

        # 新的熵，代表当前特征值的熵
        newEntropy = 0.0

        # 遍历现在有的特征的可能性
        for value in uniqueVals:
            # 在全部数据集的当前特征位置上，找到该特征值等于当前值的集合
            subDataSet = splitDataSet(dataSet=dataSet, axis=i, value=value)

            # 计算出权重
            prob = len(subDataSet) / float(len(dataSet))

            # 计算出当前特征值的熵
            newEntropy += prob * calcShannonEnt(subDataSet)

        # 计算出“信息增益”
        infoGain = baseEntropy - newEntropy

        # print('当前特征值为：' + labels[i] + '，对应的信息增益值为：' + str(infoGain))

        # 如果当前的信息增益比原来的大
        if infoGain > bestInfoGain:
            # 最好的信息增益
            bestInfoGain = infoGain
            # 新的最好的用来划分的特征值
            bestFeature = i

    # print('信息增益最大的特征为：' + labels[bestFeature])
    return bestFeature


def majorityCnt(classList):
    """
    找到次数最多的类别标签
    :param classList:
    :return:
    """
    # 用来统计标签的票数
    classCount = collections.defaultdict(int)

    # 遍历所有的标签类别
    for vote in classList:
        classCount[vote] += 1

    # 从大到小排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    # 返回次数最多的标签
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    创建决策树
    :param dataSet: 数据集
    :param labels: 特征标签
    :return:
    """
    # 拿到所有数据集的分类标签
    classList = [example[-1] for example in dataSet]

    # 统计第一个标签出现的次数，与总标签个数比较，如果相等则说明当前列表中全部都是一种标签，此时停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 计算第一行有多少个数据，如果只有一个的话说明所有的特征属性都遍历完了，剩下的一个就是类别标签
    if len(dataSet[0]) == 1:
        # 返回剩下标签中出现次数较多的那个
        return majorityCnt(classList)

    # 选择最好的划分特征，得到该特征的下标
    bestFeat = chooseBestFeatureToSplit(dataSet=dataSet, labels=labels)

    # 得到最好特征的名称
    bestFeatLabel = labels[bestFeat]

    # 使用一个字典来存储树结构，分叉处为划分的特征名称
    myTree = {bestFeatLabel: {}}

    # 将本次划分的特征值从列表中删除掉
    del(labels[bestFeat])

    # 得到当前特征标签的所有可能值
    featValues = [example[bestFeat] for example in dataSet]

    # 唯一化，去掉重复的特征值
    uniqueVals = set(featValues)

    # 遍历所有的特征值
    for value in uniqueVals:
        # 得到剩下的特征标签
        subLabels = labels[:]
        subTree = createTree(splitDataSet(dataSet=dataSet, axis=bestFeat, value=value), subLabels)
        # 递归调用，将数据集中该特征等于当前特征值的所有数据划分到当前节点下，递归调用时需要先将当前的特征去除掉
        myTree[bestFeatLabel][value] = subTree
    return myTree


def createDataSet():
    """
    创建测试的数据集
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

    # 特征对应的所有可能的情况
    labels_full = {}

    for i in range(len(labels)):
        labelList = [example[i] for example in dataSet]
        uniqueLabel = set(labelList)
        labels_full[labels[i]] = uniqueLabel

    return dataSet, labels, labels_full


def makeTreeFull(myTree, labels_full, parentClass):
    """
    将数中的不存在的特征标签进行补全，补全为父节点中出现最多的类别
    :param myTree: 生成的树
    :param labels_full: 特征的全部标签
    :param parentClass: 父节点中所含最多的类别
    :return:
    """
    # 拿到当前的根节点
    root_key = list(myTree.keys())[0]

    # 得到根节点对应的子树，也就是key对应的内容
    sub_tree = myTree[root_key]

    # 如果是叶子节点就结束
    if isinstance(sub_tree, str):
        return

    # 循环遍历全部特征标签，将不存在标签添加进去
    for label in labels_full[root_key]:
        if label not in sub_tree.keys():
            if parentClass is not None:
                sub_tree[label] = parentClass

    # 当前层对应的分类列表
    current_class = []

    # 循环遍历一下子树，找到类别最多的那个，如果此时没有分类的话就是None
    for sub_key in sub_tree.keys():
        if isinstance(sub_tree[sub_key], str):
            current_class.append(sub_tree[sub_key])

    # 找到本层出现最多的类别，作为parentLabel传递给下一次递归
    if len(current_class):
        most_class = collections.Counter(current_class).most_common(1)[0][0]
    else:
        most_class = None

    # 递归处理
    for sub_key in sub_tree.keys():
        if isinstance(sub_tree[sub_key], dict):
            makeTreeFull(myTree=sub_tree[sub_key], labels_full=labels_full, parentClass=most_class)


if __name__ == '__main__':
    myDat, labels, labels_full = createDataSet()
    # print(labels_full)
    myTree = createTree(myDat, labels)
    # print(myTree)
    makeTreeFull(myTree, labels_full, None)
    # print(myTree)
    # treeDepth = treePlotter.getTreeDepth(myTree)
    # print(treeDepth)
    # leafs = treePlotter.getNumLeafs(myTree)
    # print(leafs)
    treePlotter.createPlot(myTree)

