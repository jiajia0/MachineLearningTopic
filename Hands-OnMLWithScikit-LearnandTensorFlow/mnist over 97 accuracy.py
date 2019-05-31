# @Time    : 2019/5/31 10:15
# @Author  : Leafage
# @File    : mnist over 97 accuracy.py
# @Software: PyCharm
# @Describe: 练习题中的97%的正确率
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

if __name__=='__main__':
    mnist = fetch_mldata('MNIST original')
    X, y = mnist['data'], mnist['target']

    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)  # 返回一个六万大小的随机序列
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]  # 根据随机序列，生成随机的排序

    param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]

    knn_clf = KNeighborsClassifier()
    grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(grid_search.best_params_)
    print(grid_search.best_score_)
    y_pred = grid_search.predict(X_test)
    print(accuracy_score(y_test, y_pred))
