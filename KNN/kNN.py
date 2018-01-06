import numpy as np
import operator


def createDataSet():
    """
    测试用的数据集
    :return:
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    KNN 分类器
    :param inX: 需要分类的数据
    :param dataSet: 数据集
    :param labels: 标签
    :param k: k个最相似的数据
    :return: 返回分类结果
    """
    # 计算数据集共有多少行
    dataSetSize = dataSet.shape[0]

    # 将传入进来的inX，扩增到与整个数据集相等矩阵，并与原矩阵做差，得到与各个点之间的坐标距离
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet

    # 计算与各个点之间距离的平方
    sqDiffMat = diffMat ** 2

    # 在轴为1(这里也就是行)的方向上求和，这里计算得到的就是x方和y方的和
    sqDistances = sqDiffMat.sum(axis=1)

    # 开方得到距离
    distances = sqDistances ** 0.5

    # 将距离按照从小到大的顺序排列，最后得到下标
    sortedDistIndicies = distances.argsort()

    # 用来存放标签
    classCount = {}

    # 选择距离最小的k个数值
    for i in range(k):
        # 选择对应标签
        voteIlabel = labels[sortedDistIndicies[i]]
        # 将对应标签加一
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 进行排序，按照值进行排序，并且从大到小（逆序）排列
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    # 返回第一个元素，这里打印了一下结果
    # print(sortedClassCount)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    group, labels = createDataSet()
    print(classify0([0, 0], group, labels, 3))
