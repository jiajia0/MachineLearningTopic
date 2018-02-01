# @Time    : 2018/1/31 22:38
# @Author  : Leafage
# @File    : nba_predicted.py
# @Software: PyCharm
# @Describe: 使用2014年的NBA数据，利用决策树进行预测。


import pandas as pd
from collections import defaultdict


def load_dataset():
    """
    加载数据集
    :return:
    """
    data_filename = 'leagues_NBA_2014_games_games.csv'
    dataset = pd.read_csv(data_filename, parse_dates=['Date'])
    # 增加主场获胜的球队的特征
    dataset['HomeWin'] = dataset['VisitorPts'] < dataset['HomePts']
    # 新增主/客长上一局是否胜利的特征
    dataset['HomeLastWin'] = False
    dataset['VisitorLastWin'] = False
    #
    won_last = defaultdict(int)
    # 遍历所有的数据
    for index, row in dataset.iterrows():
        # 找到主场队伍名称
        home_team = row['Home Team']
        # 找到客场队伍名称
        visitor_team = row['Visitor Team']
        #
        row['HomeLastWin'] = won_last[home_team]

        row['VisitorLastWin'] = won_last[visitor_team]
        # 更新改行数据集
        dataset.ix[index] = row
    return dataset


if __name__ == '__main__':
    dataset = load_dataset()
    # print(dataset['HomeWin'])

