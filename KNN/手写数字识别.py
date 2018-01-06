# @Time    : 2017/12/15 13:36
# @Author  : Leafage
# @File    : 手写数字识别.py
# @Software: PyCharm

import numpy as np
from os import listdir
from KNN import kNN, imgTo01


def img2vector(filename):
    """
    将32*32的图像矩阵转化为1*1024的向量
    :param filename: 文件名
    :return:
    """
    # 制造一个空数组
    returnVect = np.zeros((1,1024))

    # 打开文件
    fr = open(filename)

    # 将矩阵转换为1*1024的
    for i in range(32):
        # 读取一行数据
        lineStr = fr.readline().strip()

        # 将每行的数字存放在数组中
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])

    return returnVect


def handwritingClassTest():
    """
    数字测试
    :return:
    """
    # 数字对应的标签，也就是数字的本身
    hwLabels = []

    # 得到目录下的所有文件名称
    trainingFileList = listdir('trainingDigits')

    # 计算共有多少个文件
    m = len(trainingFileList)

    # 构造m*1024 数组，用来存放所有的数字
    trainingMat = np.zeros((m, 1024))

    # 遍历所有的文件，将其加载到数组中
    for i in range(m):
        # 得到文件名称
        fileNameStr = trainingFileList[i]

        # 去除后面的.txt，得到有用的文件名
        fileStr = fileNameStr.split('.')[0]

        # 解析出来当前是哪个数字
        classNumStr = int(fileStr.split('_')[0])

        # 添加到标签上
        hwLabels.append(classNumStr)

        # 将文件转化为数组并存放到总的数组中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    # 得到测试文件的目录
    testFileList = listdir('testDigits')

    # 错误统计
    errorCount = 0.0

    # 测试数据的总数
    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)

        # 使用分类器得到结果
        classifierResult = kNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)

        # 打印结果
        print('使用分类器得到的结果为：%s，真实的结果为：%s' % (classifierResult, classNumStr))

        # 错误的话记录下来
        if classifierResult != classNumStr:
            errorCount += 1.0

    print('识别错误的个数为：%s' % errorCount)
    print('分类器的正确率为：%f' % (errorCount/float(mTest)))


def classifyHandwriting(filename):
    """
    将图片转化为01矩阵，然后使用分类器进行分类
    :return:
    """
    # 得到32*32的01数组
    imgTo01.picTo01(filename)

    # 得到对应名称的txt文件
    name01 = filename.split('.')[0]
    name01 = name01 + '.txt'

    # 将文件中的32*32 转化为1*1024的
    hwMat = img2vector(name01)

    # 数字对应的标签，也就是数字的本身
    hwLabels = []

    # 得到目录下的所有文件名称
    trainingFileList = listdir('trainingDigits')

    # 计算共有多少个文件
    m = len(trainingFileList)

    # 构造m*1024 数组，用来存放所有的数字
    trainingMat = np.zeros((m, 1024))

    # 遍历所有的文件，将其加载到数组中
    for i in range(m):
        # 得到文件名称
        fileNameStr = trainingFileList[i]

        # 去除后面的.txt，得到有用的文件名
        fileStr = fileNameStr.split('.')[0]

        # 解析出来当前是哪个数字
        classNumStr = int(fileStr.split('_')[0])

        # 添加到标签上
        hwLabels.append(classNumStr)

        # 将文件转化为数组并存放到总的数组中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    # 进行分类
    classifierResult = kNN.classify0(hwMat, trainingMat, hwLabels, 3)

    print('使用分类器的结果为：%d' % classifierResult )


if __name__ == '__main__':
    # testVector = img2vector('trainingDigits/0_13.txt')
    # handwritingClassTest()
    classifyHandwriting('3.png')