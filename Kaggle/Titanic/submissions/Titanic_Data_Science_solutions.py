# @Time    : 2019/8/8 8:57
# @Author  : Leafage
# @File    : Titanic_Data_Science_solutions.py
# @Software: PyCharm
# @Describe: Titanic Data Science Solutions 中的教程 , https://www.kaggle.com/startupsci/titanic-data-science-solutions

# 数据分析使用的包
import numpy as np
import pandas as pd
import random as rnd
from numpy.core.umath_tests import inner1d

# 可视化使用的包
import seaborn as sns
import matplotlib.pyplot as plt

# 机器学习的包
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Acquire data
train_df = pd.read_csv(r'../datasets/train.csv')
test_df = pd.read_csv(r'../datasets/test.csv')

# 输出信息，进行观察数据

# print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# 可视化年龄和生存的数据
# g = sns.FacetGrid(train_df, col='Survived')
# g.map(plt.hist, 'Age', bins=20)

# 可视化存活、年龄、Pclass之间的关系
# g = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# g.map(plt.hist, 'Age', alpha=.5, bins=20)
# g.add_legend()

# Pclass Sex 和 Embarked 和 存活 的关系
# g = sns.FacetGrid(train_df, col='Embarked', size=2.2, aspect=1.6)
# g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# g.add_legend()

# 验证 登港口 性别和 存活率的关系
# g = sns.FacetGrid(train_df, col='Survived', row='Embarked', size=2.2, aspect=1.6)
# g.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
# g.add_legend()
# plt.show()

# 删除Ticket和Cabin特征
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

# 提取出姓名中的称呼
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# 绘制生存和Title以及Age的图 , 进行观察
# g = sns.FacetGrid(train_df, col='Survived', row='Title', size=2.2, aspect=1.6)
# g.map(plt.hist, 'Age', bins=20)
# g.add_legend()
# plt.show()

# 使用常见的name替代这些称呼
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

# 把categorical 转换为 ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# print(train_df.head())

# 删除Name和Passengerld
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

# 把male转换为0， female转换为1
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

# print(train_df.head())

# 绘制Pclass、Sex、Age的图，观察之间的关系，用来补全Age
# grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()
# plt.show()

# 创建一个2*3的empty array， 存储Sex ： 0 ， 1 Pclass ：1，2，3
guess_ages = np.zeros((2, 3))

for dataset in combine:
    for i in range(0, 2):  # Sex
        for j in range(0, 3):  # Pclass
            # eg：Sex ： 0 ， Pclass：2 的Age，并且确实空值
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            # 找对对应的空值，然后进行补全
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j]
    dataset['Age'] = dataset['Age'].astype(int)

# 把Age划分为5个等级
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
rs = train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand',
                                                                                               ascending=True)

# 转换为ordinals
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

# 删除AgeBand
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

# 统计家庭的总人数
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

rs = train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived',
                                                                                                     ascending=False)

# 统计是独自一人的特征
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

rs = train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

# 删除Parch、Sibsp、FamilySize ，只保留 isAlone
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

# 添加新特征
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

# 使用频率最高的代替缺失值
freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

rs = train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                                 ascending=False)
# 转化为数值型
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# 用中位数补全缺失值
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

# 划分为4个等级
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
rs = train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',
                                                                                                 ascending=True)

# 转化为ordinal
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

# 至此，所有数据已全部转化为数值型
# print(train_df.head())

# 使用模型进行训练预测
X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()

# 使用Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# print(acc_log)  # 81.26

# 计算feature之间的相关性
coeff_df = pd.DataFrame(train_df.columns.delete(0))  # 删除Survived这一列
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
rs = coeff_df.sort_values(by='Correlation', ascending=False)  # 按照相关性从上往下排序

# acc每次计算均有波动
# 使用SVM
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
# print(acc_svc)  # 83.5

# 使用KNN， n=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
# print(acc_knn)  # 84.06

# 使用高斯朴素贝叶斯
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
# print(acc_gaussian)  # 76.88

# 使用感知机
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
# print(acc_perceptron)  # 78.79

# 使用Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
# print(acc_linear_svc)  # 79.46

# 使用SGD
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
# print(acc_sgd)  # 76.66

# 使用决策树
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
# print(acc_decision_tree)  # 86.64

# 使用随机森林
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
# print(acc_random_forest)  # 86.64

# Model evaluation
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
rs = models.sort_values(by='Score', ascending=False)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred  # random forest 的预测结果
    })

# Top 74 %
submission.to_csv('../datasets/my_submission.csv', index=False)
