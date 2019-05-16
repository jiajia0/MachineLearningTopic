# @Time    : 2019/4/25 19:47
# @Author  : Leafage
# @File    : 02_end_to_end_machine_learning_project.py
# @Software: PyCharm
# @Describe: 书中第二章内容
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib


# 能够显示中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.serif'] = ['SimHei']


HOUSING_PATH = 'housing.csv'


def load_housing_data():
    return pd.read_csv(HOUSING_PATH)


if __name__=='__main__':
    housing = load_housing_data()
    #housing.hist(bins=50,figsize=(20,15))
    #plt.show()
    print(housing['ocean_proximity'].value_counts())