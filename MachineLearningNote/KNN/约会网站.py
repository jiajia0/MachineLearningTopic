import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from KNN import kNN


def file2matrix(filename):
    """
    将文件转换为特征矩阵和对应的标签
    :param filename:
    :return:
    """
    # 打开文件
    fr = open(filename)
    # 按照行进行读取
    arrayOLines = fr.readlines()
    # 计算出总行数，为1000行
    numbserOfLines = len(arrayOLines)
    # 创建一个1000行的0矩阵
    returnMat = np.zeros((numbserOfLines, 3))
    classLabelVector = []
    index = 0

    for line in arrayOLines:
        # 去除前后空格
        line = line.strip()
        # 按照\t将每行内容分割开来
        listFormLine = line.split('\t')
        # 将得到前三项的内容填充到对应的行数矩阵上
        returnMat[index, :] = listFormLine[0:3]
        # 按照喜欢程度进行添加标签
        # classLabelVector.append(int(listFormLine[-1]))
        if listFormLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        if listFormLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        if listFormLine[-1] == 'didntLike':
            classLabelVector.append(1)
        # 行数索引加一
        index += 1

    return returnMat, classLabelVector


def dataShow(datingDataMat, datingLabels):
    """
    将矩阵可视化，画出每年获得的飞行常客里程数和玩视频游戏所消耗时间百分比的图示
    :param datingDataMat:
    :param datingLabels:
    :return:
    """

    # 将对应的标签转换为颜色
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append("b")  # 黑色不喜欢
        if i == 2:
            LabelsColors.append("y")  # 黄色一般喜欢
        if i == 3:
            LabelsColors.append("r")  # 红色很喜欢

    # 能够显示中文
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['font.serif'] = ['SimHei']

    # matplotlib的图像都位于Figure对象中，设置大小和分辨率
    fig = plt.figure(figsize=(12, 7), dpi=80)

    # 第一个图，用来显示每年获得的飞行常客里程数和玩视频游戏所消耗时间百分比
    ax1 = fig.add_subplot(1, 2, 1)
    # 设置x标签
    ax1.set_xlabel('每年获得飞行常客里程数')
    # 设置y标签
    ax1.set_ylabel('玩视频游戏所消耗时间百分比')
    # 添加数据，第一列就是飞行里程数，第二列是玩视频游戏的时间
    ax1.scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=LabelsColors,
                s=(15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels)), marker='o')

    didnt_Like = ax1.scatter([], [], c='b', s=15)
    smallDoses = ax1.scatter([], [], c='y', s=30)
    largeDoses = ax1.scatter([], [], c='r', s=45)
    # 添加标签
    ax1.legend((didnt_Like, smallDoses, largeDoses), ('不喜欢', '魅力一般', '极具魅力'), loc='best')

    # 另一种作图思路，作出玩视频游戏所消耗百分比(x)和每周消耗冰淇淋公斤数(y)的图
    ax2 = fig.add_subplot(1, 2, 2)
    # 设置x标签
    ax2.set_xlabel('玩视频游戏所消耗百分比')
    # 设置y标签
    ax2.set_ylabel('每周消耗冰淇淋公斤数')

    # 遍历标签，划分出三种类别的坐标
    ice_didntLike = []  # 不喜欢类型的冰淇淋公斤数
    game_didntLike = []  # 不喜欢类型的玩游戏时间

    ice_smallDoses = []  # 一般魅力的冰淇淋公斤数
    game_smallDoses = []  # 一般魅力你的玩游戏时间

    ice_largeDoses = []  # 极大魅力的冰淇淋公斤数
    game_largeDoses = []  # 极大魅力的玩游戏时间

    num = 0  # 索引

    for i in datingLabels:
        if i == 1:  # 不喜欢
            ice_didntLike.append(datingDataMat[num, 2])
            game_didntLike.append(datingDataMat[num, 1])
        if i == 2:  # 一般魅力
            ice_smallDoses.append(datingDataMat[num, 2])
            game_smallDoses.append(datingDataMat[num, 1])
        if i == 3:  # 极大魅力
            ice_largeDoses.append(datingDataMat[num, 2])
            game_largeDoses.append(datingDataMat[num, 1])

        num += 1

    # 极大魅力绘制
    largeDoses = ax2.scatter(x=game_largeDoses, y=ice_largeDoses, s=45, c='r')
    # 一般魅力绘制
    smallDoses = ax2.scatter(x=game_smallDoses, y=ice_smallDoses, s=30, c='y')
    # 不喜欢绘制
    didntLike = ax2.scatter(x=game_didntLike, y=ice_didntLike, s=15, c='b')

    # 添加标签
    ax2.legend((didntLike, smallDoses, largeDoses), ('不喜欢', '魅力一般', '极具魅力'), loc='best')

    # 自动调整间距防止标签超过范围
    plt.tight_layout()
    plt.show()


def autoNorm(dataSet):
    """
    数值归一化，将所有的特征值控制在0到1之间
    :param dataSet:
    :return:
    """
    # 按照axis=0的方向计算出最大值（也就是找到每列中的最大值）/ 或最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)

    # 计算出差值
    ranges = maxVals - minVals

    # 创建一个用0填充的数组
    normDataSet = np.zeros(shape=(np.shape(dataSet)))

    # 得到总行数
    m = dataSet.shape[0]

    # 将最小值扩充到与原矩阵相等的样子，也就是1000*3的样子，然后相减
    normDataSet = dataSet - np.tile(minVals, (m, 1))

    # 再次相除
    normDataSet = normDataSet/np.tile(ranges, (m, 1))

    return normDataSet, ranges, minVals


def datingClassTest():
    """
    测试分类器的准确率
    :return:
    """
    # 一个比率，与总数据相乘应该得到100，就是我们需要测试的100条数据
    hoRatio = 0.10

    # 拿到数据
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')

    # 数据归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)

    # 计算出总行数
    m = normMat.shape[0]

    # 得到总的100
    numTestVecs = int(m*hoRatio)

    # 错误率初始化为0
    errorCount = 0.0

    # 使用kNN进行分类验证前一百个数据
    for i in range(numTestVecs):
        # 使用原来的分类器
        classifierResult = kNN.classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)

        # 打印出预测结果和真实结果
        print('使用分类器分类得到的结果为：%d ,当前数据的真实结果为：%d 。' % (classifierResult, datingLabels[i]))

        # 如果不对的话错误值加一
        if classifierResult != datingLabels[i]:
            errorCount += 1.0

    # 打印出错误率
    print('分类器的错误率为：' + str(errorCount/float(numTestVecs)))


def classifyPerson():
    """
    输入内容进行预测
    :return:
    """
    # 结果列表，不喜欢、一般魅力、极具魅力
    resultList = ['not at all', 'in small doses', 'in large doses']

    # 输入玩视频游戏所消耗时间的百分比
    percentTats = float(input('玩视频游戏所消耗时间的百分比：'))

    # 输入每年获得的飞行常客里程数
    ffMiles = float(input('每年获得的飞行常客里程数：'))

    # 输入每周消费的冰淇淋公斤数
    iceCream = float(input('每周消费的冰淇淋公斤数：'))

    # 拿到数据
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')

    # 数据归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)

    # 将输入的数据转换为数组
    inArr = np.array([ffMiles, percentTats, iceCream])

    # 测试数据归一化
    normInArr = (inArr - minVals) / ranges

    # 进行分类
    classifierResult = kNN.classify0(normInArr, normMat, datingLabels, 3)

    print('你对此人的感觉可能是：' + resultList[classifierResult - 1])


if __name__ == '__main__':
    # 拿到数据
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')

    # 数据可视化
    # dataShow(datingDataMat, datingLabels)

    # 检测分类器的准确性
    # datingClassTest()

    # 进行分类
    classifyPerson()
