# @Time    : 2017/12/15 18:11
# @Author  : Leafage
# @File    : imgTo01.py
# @Software: PyCharm

from PIL import Image
import matplotlib.pylab as plt
import numpy as np


def picTo01(filename):
    """
    将图片转化为32*32像素的文件，用0 1表示
    :param filename:
    :return:
    """
    # 打开图片
    img = Image.open(filename).convert('RGBA')

    # 得到图片的像素值
    raw_data = img.load()

    # 将其降噪并转化为黑白两色
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if raw_data[x, y][0] < 90:
                raw_data[x, y] = (0, 0, 0, 255)

    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if raw_data[x, y][1] < 136:
                raw_data[x, y] = (0, 0, 0, 255)

    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if raw_data[x, y][2] > 0:
                raw_data[x, y] = (255, 255, 255, 255)

    # 设置为32*32的大小
    img = img.resize((32, 32), Image.LANCZOS)

    # 进行保存，方便查看
    img.save('test.png')

    # 得到像素数组，为(32,32,4)
    array = plt.array(img)
    # array = np.asarray(img)
    # 按照公式将其转为01, 公式： 0.299 * R + 0.587 * G + 0.114 * B

    gray_array = np.zeros((32, 32))

    # 行数
    for x in range(array.shape[0]):
        # 列数
        for y in range(array.shape[1]):
            # 计算灰度，若为255则白色，数值越小越接近黑色
            gary = 0.299 * array[x][y][0] + 0.587 * array[x][y][1] + 0.114 * array[x][y][2]

            # 设置一个阙值，记为0
            if gary == 255:
                gray_array[x][y] = 0
            else:
                # 否则认为是黑色，记为1
                gray_array[x][y] = 1

    # 得到对应名称的txt文件
    name01 = filename.split('.')[0]
    name01 = name01 + '.txt'

    # 保存到文件中
    np.savetxt(name01, gray_array, fmt='%d', delimiter='')


if __name__ == '__main__':
    picTo01('3.png')

