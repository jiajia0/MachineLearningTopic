# @Time    : 2019/5/31 11:06
# @Author  : Leafage
# @File    : kaggle_titanic.py
# @Software: PyCharm
# @Describe: Kaggle Titanic
import os
import pandas as pd
TITANIC_PATH = os.path.join('datasets', 'titanic')


# 使用pd读取数据
def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)


train_data = load_titanic_data('train.csv')
test_data = load_titanic_data('test.csv')
# print(train_data.info())  # 查看相关数据是否缺失
# print(train_data.describe())  # 查看数据情况
print(train_data['Survived'].value_counts())  # 查看存活分类 , 其他分类也可以进行查看观测


