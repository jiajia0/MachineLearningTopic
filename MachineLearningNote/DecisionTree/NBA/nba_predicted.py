# @Time    : 2018/1/31 22:38
# @Author  : Leafage
# @File    : nba_predicted.py
# @Software: PyCharm
# @Describe: 使用2014年的NBA数据，利用决策树进行预测。


import pandas as pd


def load_dataset():
    """
    加载数据及
    :return:
    """
    data_filename = 'leagues_NBA_2014_games_games.csv'
    dataset = pd.read_csv(data_filename)
    print(dataset.ix[:5])


if __name__ == '__main__':
    load_dataset()
