# @Time    : 2019/5/15 10:06
# @Author  : Leafage
# @File    : 03_mnist.py
# @Software: PyCharm
# @Describe:

from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

mnist = fetch_mldata('MNIST original')
X, y = mnist['data'], mnist['target']

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis('off')
#plt.show()
#print(y[36000])
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)  # 返回一个六万大小的随机序列
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index] # 根据随机序列，生成随机的排序

# 二分类训练
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
result = sgd_clf.predict([some_digit])

skfolds = StratifiedKFold(n_splits=3, random_state=42)

'''
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
'''

result = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

result = confusion_matrix(y_train_5, y_train_pred)

print(result)

print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))