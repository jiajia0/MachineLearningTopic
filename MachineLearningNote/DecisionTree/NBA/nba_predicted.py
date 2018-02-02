# @Time    : 2018/1/31 22:38
# @Author  : Leafage
# @File    : nba_predicted.py
# @Software: PyCharm
# @Describe: 使用2014年的NBA数据，利用决策树进行预测。


import pandas as pd
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


def load_dataset():
    """
    加载数据集
    :return:
    """
    data_filename = 'leagues_NBA_2014_games_games.csv'
    dataset = pd.read_csv(data_filename, parse_dates=['Date'])
    # 增加主场获胜的球队的特征
    dataset['HomeWin'] = dataset['VisitorPts'] < dataset['HomePts']
    # 新增主/客场队的上一次比赛是否胜利的特征
    dataset['HomeLastWin'] = False
    dataset['VisitorLastWin'] = False
    # key：队伍名称，value：上次比赛是否胜利
    won_last = defaultdict(bool)
    # 遍历所有的数据，记录每一次比赛中的各队在上次比赛中是否胜利
    for index, row in dataset.iterrows():
        # 找到主场队伍名称
        home_team = row['Home Team']
        # 找到客场队伍名称
        visitor_team = row['Visitor Team']
        # 更新上次主场队是否胜利
        row['HomeLastWin'] = won_last[home_team]
        # 更新上次客场队是否胜利
        row['VisitorLastWin'] = won_last[visitor_team]
        # 更新改行数据集
        dataset.ix[index] = row
        # 设置本次主场队是否胜利
        won_last[home_team] = row['HomeWin']
        # 设置本次客场队是否胜利
        won_last[visitor_team] = not row['HomeWin']
    return dataset


def createTree(dataset):
    """
    创建决策树
    :param dataset:
    :return:
    """

    clf = DecisionTreeClassifier(random_state=14)

    x_previouswins = dataset[['HomeLastWin', 'VisitorLastWin']].values

    y_true = dataset['HomeWin']

    scores = cross_val_score(clf, x_previouswins, y_true, scoring='accuracy')

    print("Using just the last result from the home and visitor teams")
    print(np.mean(scores) * 100)


if __name__ == '__main__':
    dataset = load_dataset()
    createTree(dataset)
