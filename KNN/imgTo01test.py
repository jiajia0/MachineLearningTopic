# @Time    : 2017/12/15 18:11
# @Author  : Leafage
# @Site    : 
# @File    : 灰度化.py
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
    img = Image.open(filename)

    # 得到图片的像素值
    raw_data = img.getdata()

    # 用来保存灰度图片的数据
    gray_data = []

    # 新创建一个图片，用来保存灰度处理后的图片
    gray_image = Image.new('L', img.size)

    # 将其转化为黑白两色
    for rgb in raw_data:
        value = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        # 阙值，可以改变
        if value < 140:
            # 写入黑色
            gray_data.append(0)
        else:
            # 写入白色
            gray_data.append(255)

    # 写入到图片中
    gray_image.putdata(gray_data)

    # 设置像素大小
    gray_image = gray_image.resize((32, 32), Image.ANTIALIAS)

    # 缩略为32*32
    # gray_image.thumbnail((32, 32))

    # 保存图片
    gray_image.save('灰度化并压缩处理后的图片.png')

    # 得到像素数组
    array = plt.array(gray_image)

    np.savetxt('ce.txt', array, fmt='%d', delimiter='')

    # 去掉多余的，使其为32*32
    # array = array[0:32, 0:32]

    # 如果行数少于32，则将其补充到32
    # if array.shape[0] < 32:
    #     array = np.row_stack((array, np.zeros((32-array.shape[0], 32))))
    #
    # # 如果列数少于32，则将其补充到32
    # if array.shape[1] < 32:
    #     array = np.column_stack((array, np.zeros((32, 32-array.shape[1]))))

    # 将255附近全部用0表示，非255附近全部用1表示
    array[array == 255] = 0
    array[array != 0] = 1

    # 得到对应名称的txt文件
    name01 = filename.split('.')[0]
    name01 = name01 + '.txt'

    # 保存到文件中
    np.savetxt(name01, array, fmt='%d', delimiter='')


if __name__ == '__main__':
    picTo01('2.png')

